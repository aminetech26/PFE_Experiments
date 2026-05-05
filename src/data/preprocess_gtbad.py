import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")


class PVDataPreprocessor:
    """
    Full preprocessing pipeline as described in:
    "GTBAD: GVSAO-Transformer-BiLSTM-based time-series anomaly detection ..."
    """

    def __init__(
        self,
        window_len=10,
        stride=1,
        power_col="pdc1",
        corr_threshold=0.85,
    ):
        """
        Args:
            window_len (int): sliding window length S (default 10)
            stride (int): sliding stride (default 1)
            power_col (str): column name used as 'power' for correlation priority
            corr_threshold (float): absolute Pearson correlation threshold for feature selection
        """
        self.window_len = window_len
        self.stride = stride
        self.power_col = power_col
        self.corr_threshold = corr_threshold

        self.scaler = MinMaxScaler()
        self.selected_features = None
        self.fitted = False

    def fit_transform(self, df, timestamp_col="timestamp"):
        """
        Apply the entire preprocessing pipeline and return
        - X: input windows (features + time encodings), shape (n_samples, 3*window_len, n_input_features)
        - y: target windows (selected numeric features only), shape (n_samples, 3*window_len, n_selected)
        - mask: training mask (0=exclude from loss), shape (n_samples,)
        - df_processed: the cleaned, scaled, smoothed, feature-selected DataFrame
        """
        if self.fitted:
            raise RuntimeError(
                "Preprocessor already fitted. Create a new instance or implement a separate transform method."
            )

        # 1. Copy and handle missing values (median fill)
        df_clean = df.copy()
        numeric_feats = [c for c in df.columns if c not in [timestamp_col, "anomaly"]]
        orig_missing = df_clean[numeric_feats].isna()  # record original missingness
        for col in numeric_feats:
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)

        # 2. Min-Max scaling
        scaled_data = self.scaler.fit_transform(df_clean[numeric_feats])
        df_scaled = pd.DataFrame(scaled_data, columns=numeric_feats)

        # 3. Pearson correlation-based feature selection
        corr_matrix = df_scaled.corr().abs()
        to_drop = set()
        for i in range(len(numeric_feats)):
            for j in range(i + 1, len(numeric_feats)):
                col_i = numeric_feats[i]
                col_j = numeric_feats[j]
                if col_i in to_drop or col_j in to_drop:
                    continue
                if corr_matrix.loc[col_i, col_j] >= self.corr_threshold:
                    # decide which to keep
                    keep_i = self._feature_priority(col_i, col_j, df_scaled, orig_missing)
                    if keep_i == col_i:
                        to_drop.add(col_j)
                    else:
                        to_drop.add(col_i)
        self.selected_features = [c for c in numeric_feats if c not in to_drop]
        print(f"Selected {len(self.selected_features)} features: {self.selected_features}")

        # 4. S-G smoothing on selected features
        df_smooth = df_scaled[self.selected_features].copy()
        for col in self.selected_features:
            df_smooth[col] = savgol_filter(
                df_smooth[col],
                window_length=self.window_len,
                polyorder=3,
                mode="nearest",
            )
        df_processed = df_smooth.copy()

        # 5. Sliding window construction
        n_total = len(df_processed)
        # build full sequence of selected features (n_total, n_selected)
        feat_array = df_processed.values  # shape (T, F)
        windows = []
        target_windows = []
        mask_values = []

        start = self.window_len - 1
        for i in range(start, n_total):
            # current window: i-window_len+1 : i
            cur_win = feat_array[i - self.window_len + 1 : i + 1, :]
            windows.append(cur_win)
            # target is exactly the current window (for reconstruction)
            target_windows.append(cur_win)

            # training mask: exclude if any of the original values in this window were missing
            # (we use the original missingness recorded before median fill)
            idx_range = range(i - self.window_len + 1, i + 1)
            if orig_missing.iloc[idx_range, :].any().any():
                mask_values.append(0)
            else:
                mask_values.append(1)

        X_numeric = np.array(windows)  # (n_samples, window_len, n_selected)
        y_target = np.array(target_windows)  # (n_samples, window_len, n_selected)
        mask = np.array(mask_values)  # (n_samples,)
        print(f"Generated {X_numeric.shape[0]} samples with mask sum={mask.sum()}")

        # 6. Add time encodings (hour, dayofweek, holiday)
        # Need to extract timestamps for each time step in the window.
        # We'll use the original timestamps from df.
        timestamps = pd.to_datetime(df[timestamp_col]).values

        # For each sample, the times for the positions:
        # Positions 0-9: cur_win (i-9 .. i)
        n_samples = X_numeric.shape[0]
        time_feat_dim = 24 + 7 + 1  # hour one-hot + dayofweek one-hot + holiday binary
        time_feats = np.zeros((n_samples, self.window_len, time_feat_dim))

        # Simple holiday detection: none (can be extended)
        is_holiday = np.zeros(n_total, dtype=bool)

        for idx, i in enumerate(range(start, n_total)):
            # current window indices
            cur_idx = np.arange(i - self.window_len + 1, i + 1)
            for j, t_idx in enumerate(cur_idx):
                ts = timestamps[t_idx]
                hour = ts.hour
                dow = ts.dayofweek  # Monday=0, Sunday=6
                hol = 1 if is_holiday[t_idx] else 0

                time_feats[idx, j, hour] = 1.0  # hour one-hot
                time_feats[idx, j, 24 + dow] = 1.0  # day-of-week one-hot
                time_feats[idx, j, -1] = hol  # holiday flag

        # Concatenate numerical and time features
        X_full = np.concatenate(
            [X_numeric, time_feats], axis=2
        )  # (n_samples, window_len, n_selected + time_feat_dim)
        self.fitted = True
        self._mask = mask
        return X_full, y_target, mask, df_processed

    def _feature_priority(self, col_a, col_b, df_scaled, orig_missing):
        """Return the column to keep based on: 1) correlation with power, 2) missing rate, 3) variance."""
        power_corr = df_scaled.corr()[self.power_col].abs()
        corr_a = power_corr.get(col_a, 0)
        corr_b = power_corr.get(col_b, 0)
        if corr_a > corr_b:
            return col_a
        if corr_b > corr_a:
            return col_b
        # missing rate (lower is better)
        miss_a = orig_missing[col_a].mean()
        miss_b = orig_missing[col_b].mean()
        if miss_a < miss_b:
            return col_a
        if miss_b < miss_a:
            return col_b
        # variance (larger is better)
        var_a = df_scaled[col_a].var()
        var_b = df_scaled[col_b].var()
        return col_a if var_a >= var_b else col_b

    def get_mask(self):
        if not self.fitted:
            raise RuntimeError("Must fit before accessing mask.")
        return self._mask
