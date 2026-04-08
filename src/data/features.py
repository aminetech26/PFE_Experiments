"""
Feature engineering pipeline for PV fault detection.

Stages:
  1. Physics-based features (domain knowledge)
  2. Statistical window features (sliding window)
  3. Signal-based features (FFT, CEEMDAN, Wavelet)
  4. TSFRESH automated features (top-N selection)

All transforms must be fitted on TRAINING data only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from scipy import signal
from scipy.fft import rfft, rfftfreq


# ============================================================================
# PHYSICS-BASED FEATURES
# ============================================================================

def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add domain-knowledge features.
    Column names assume La Réunion / merged dataset naming conventions.
    Adjust column references for Mendeley if needed.
    """
    out = df.copy()

    # --- Power-based ---
    if "Pg" in out.columns and "GTI" in out.columns:
        # Performance Ratio: actual power / expected from irradiance
        # P_STC is in watts; GTI normalized by 1000 W/m² (STC reference)
        # Requires knowing P_STC — use median of clean periods as proxy if unknown
        gti_safe = out["GTI"].replace(0, np.nan)
        out["performance_ratio"] = out["Pg"] / (gti_safe / 1000.0)

    if "Pg" in out.columns:
        out["dP_dt"] = out["Pg"].diff().fillna(0)

    if "Vg" in out.columns:
        out["dV_dt"] = out["Vg"].diff().fillna(0)

    if "Ig" in out.columns:
        out["dI_dt"] = out["Ig"].diff().fillna(0)

    # --- Thermal stress ---
    if "TA" in out.columns and "TPV" in out.columns:
        out["delta_temp"] = out["TPV"] - out["TA"]

    # --- Normalized voltage (remove diurnal trend) ---
    if "Vg" in out.columns:
        rolling_mean = out["Vg"].rolling(window=100, min_periods=1, center=True).mean()
        out["Vg_normalized"] = out["Vg"] / rolling_mean.replace(0, np.nan)

    # --- Mendeley-specific: Fill Factor proxy ---
    if all(c in out.columns for c in ["Vpv", "Ipv", "Vdc"]):
        out["Ppv"] = out["Vpv"] * out["Ipv"]
        out["dPpv_dt"] = out["Ppv"].diff().fillna(0)
        # AC current imbalance (3-phase)
        if all(c in out.columns for c in ["ia", "ib", "ic"]):
            currents = out[["ia", "ib", "ic"]]
            out["current_imbalance"] = currents.max(axis=1) - currents.min(axis=1)
            out["current_std"] = currents.std(axis=1)

    return out


# ============================================================================
# SLIDING WINDOW FEATURES
# ============================================================================

def create_sliding_windows(
    df: pd.DataFrame,
    window_size: int,
    step: int,
    feature_cols: list[str],
    label_col: str | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Convert a time-series DataFrame into overlapping windows.
    
    ⚠️ TODO: Add segment_col parameter to respect segment boundaries!
    Current implementation can create windows that span overnight gaps
    and multi-day gaps, violating the constraint in EDA that windows
    should only contain rows with the same segment_id.

    Args:
        df: DataFrame sorted by time
        window_size: number of timesteps in each window
        step: stride between windows
        feature_cols: columns to include
        label_col: if provided, use last label in window

    Returns:
        X: shape (n_windows, window_size, n_features)
        y: shape (n_windows,) or None
    """
    X_list, y_list = [], []
    values = df[feature_cols].values.astype(np.float32)
    labels = df[label_col].values if label_col else None

    for start in range(0, len(df) - window_size + 1, step):
        end = start + window_size
        X_list.append(values[start:end])
        if labels is not None:
            # Use majority vote within window (robust to label noise)
            window_labels = labels[start:end]
            y_list.append(np.bincount(window_labels.astype(int)).argmax())

    X = np.stack(X_list) if X_list else np.empty((0, window_size, len(feature_cols)))
    y = np.array(y_list) if y_list else None
    return X, y


def extract_window_statistics(X: np.ndarray) -> np.ndarray:
    """
    Collapse windows to statistical features.
    Input: (n_windows, window_size, n_features)
    Output: (n_windows, n_features * 8) — mean, std, min, max, skew, kurt, energy, zcr

    Use for classical ML models (non-sequential input).
    """
    n, w, f = X.shape
    features = []
    for i in range(f):
        ch = X[:, :, i]  # (n, w)
        features.append(ch.mean(axis=1))
        features.append(ch.std(axis=1))
        features.append(ch.min(axis=1))
        features.append(ch.max(axis=1))
        # Skewness
        from scipy.stats import skew, kurtosis
        features.append(skew(ch, axis=1))
        features.append(kurtosis(ch, axis=1))
        # Energy (RMS)
        features.append(np.sqrt((ch ** 2).mean(axis=1)))
        # Zero-crossing rate
        zcr = ((np.diff(np.sign(ch), axis=1) != 0).sum(axis=1)) / (w - 1)
        features.append(zcr)

    return np.column_stack(features).astype(np.float32)


# ============================================================================
# SPECTRAL FEATURES (FFT)
# ============================================================================

def extract_fft_features(X: np.ndarray, sample_rate: float = 1.0, n_top_freqs: int = 5) -> np.ndarray:
    """
    Extract top-N spectral power features from each window and channel.
    Input: (n_windows, window_size, n_features)
    Output: (n_windows, n_features * n_top_freqs)
    """
    n, w, f = X.shape
    results = []
    for i in range(f):
        ch = X[:, :, i]  # (n, w)
        fft_mag = np.abs(rfft(ch, axis=1))[:, 1:]  # drop DC
        # Take top-N magnitudes
        idx = np.argsort(fft_mag, axis=1)[:, -n_top_freqs:]
        top_mags = np.take_along_axis(fft_mag, idx, axis=1)
        results.append(top_mags)

    return np.concatenate(results, axis=1).astype(np.float32)


# ============================================================================
# WAVELET DENOISING (for La Réunion / Sonalgaz)
# ============================================================================

def wavelet_denoise_series(series: np.ndarray, wavelet: str = "db4", level: int = 4) -> np.ndarray:
    """
    Denoise a 1D time series using wavelet thresholding.
    Keeps approximation coefficients, thresholds detail coefficients.
    """
    import pywt
    coeffs = pywt.wavedec(series, wavelet, level=level)
    # Universal threshold
    threshold = np.sqrt(2 * np.log(len(series))) * np.median(np.abs(coeffs[-1])) / 0.6745
    coeffs_thresh = [pywt.threshold(c, threshold, mode="soft") for c in coeffs]
    return pywt.waverec(coeffs_thresh, wavelet)[: len(series)]


def wavelet_energy_features(series: np.ndarray, wavelet: str = "db4", level: int = 4) -> np.ndarray:
    """Extract energy per decomposition level as features."""
    import pywt
    coeffs = pywt.wavedec(series, wavelet, level=level)
    energies = np.array([(c ** 2).sum() for c in coeffs])
    return energies / (energies.sum() + 1e-10)  # normalize


# ============================================================================
# CEEMDAN (for Mendeley 10kHz data)
# ============================================================================

def ceemdan_features(series: np.ndarray, max_imfs: int = 5) -> np.ndarray:
    """
    Decompose signal using CEEMDAN (Complete EEMD with Adaptive Noise).
    Returns mean and energy of each IMF as features.
    Input: 1D array of raw signal
    Output: 1D feature vector of length (max_imfs * 2)
    """
    try:
        from emd import sift
        imfs = sift.ensemble_sift(series, max_imfs=max_imfs)
    except ImportError:
        logger.warning("EMD-signal not installed, using zero-padded features")
        return np.zeros(max_imfs * 2, dtype=np.float32)

    n_found = imfs.shape[1] if imfs.ndim > 1 else 1
    feats = []
    for j in range(min(max_imfs, n_found)):
        imf = imfs[:, j] if imfs.ndim > 1 else imfs
        feats.append(np.mean(imf))
        feats.append(np.sqrt((imf ** 2).mean()))  # RMS energy

    # Pad if fewer IMFs found
    while len(feats) < max_imfs * 2:
        feats.append(0.0)

    return np.array(feats[:max_imfs * 2], dtype=np.float32)


# ============================================================================
# GRAMIAN ANGULAR SUMMATION FIELD (classification bonus)
# ============================================================================

def to_gramian_gasf(series: np.ndarray) -> np.ndarray:
    """
    Convert a 1D time series to a Gramian Angular Summation Field (GASF) image.
    Useful for CNN-based classification experiments.
    """
    # Min-max normalize to [-1, 1]
    s_min, s_max = series.min(), series.max()
    if s_max == s_min:
        normed = np.zeros_like(series)
    else:
        normed = 2 * (series - s_min) / (s_max - s_min) - 1

    normed = np.clip(normed, -1, 1)
    phi = np.arccos(normed)
    gasf = np.cos(phi[:, None] + phi[None, :])
    return gasf.astype(np.float32)


# ============================================================================
# PREPROCESSING TRANSFORMER (fits on train, transforms train/val/test)
# ============================================================================

class FeaturePreprocessor:
    """
    Stateful preprocessor that must be fitted only on training data.

    Usage:
        prep = FeaturePreprocessor(feature_cols=...)
        X_train = prep.fit_transform(df_train)
        X_val   = prep.transform(df_val)
        X_test  = prep.transform(df_test)    # ONLY ONCE, at the end
    """

    def __init__(self, feature_cols: list[str], scale: bool = True):
        self.feature_cols = feature_cols
        self.scale = scale
        self._scaler = None
        self._fitted = False

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        from sklearn.preprocessing import RobustScaler
        X = df[self.feature_cols].values.astype(np.float32)
        if self.scale:
            self._scaler = RobustScaler()  # Robust to outliers from fault events
            X = self._scaler.fit_transform(X)
        self._fitted = True
        return X

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit_transform on training data first!")
        X = df[self.feature_cols].values.astype(np.float32)
        if self.scale and self._scaler is not None:
            X = self._scaler.transform(X)
        return X

    def save(self, path: str) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "FeaturePreprocessor":
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
