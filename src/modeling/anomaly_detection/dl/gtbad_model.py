import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # standard PE
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        Returns x with added PE.
        """
        seq_len = x.size(1)
        standard_pe = self.pe[:, :seq_len, :]
        return x + standard_pe


class GTBADModel(nn.Module):
    """
    GTBAD: GVSAO-Transformer-BiLSTM anomaly detection model.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        d_model=64,
        nhead=2,
        num_encoder_layers=3,
        lstm_hidden=32,
        dropout=0.1,
    ):
        """
        Args:
            input_dim (int): dimension after concatenating numerical features and time encodings
            output_dim (int): number of selected numerical features to reconstruct
            d_model (int): transformer model dimension
            nhead (int): number of attention heads
            num_encoder_layers (int): number of transformer encoder layers
            lstm_hidden (int): hidden size of each LSTM direction
            dropout (float): dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model

        self.input_project = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        self.bilstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.fc_out = nn.Linear(lstm_hidden * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        Returns: reconstructed numerical features, shape (batch, seq_len, output_dim)
        """
        # project to d_model
        x_proj = self.input_project(x)  # (B, L, d_model)
        # add positional encoding
        x_pe = self.pos_encoder(x_proj)  # (B, L, d_model)
        # transformer encoder
        enc_out = self.transformer_encoder(x_pe)  # (B, L, d_model)
        # BiLSTM decoder
        lstm_out, _ = self.bilstm(enc_out)  # (B, L, 2*lstm_hidden)
        # dropout and output projection
        lstm_out = self.dropout(lstm_out)
        recon = self.fc_out(lstm_out)  # (B, L, output_dim)
        return recon


def reconstruction_error(x_true, x_pred, mask=None):
    """
    Compute MSE reconstruction error per sample (sum over features and time steps).
    If mask is provided, loss for masked samples (mask=0) is set to zero.
    """
    # x_true and x_pred: (batch, seq_len, output_dim)
    error = torch.sum((x_true - x_pred) ** 2, dim=(1, 2))  # (batch,)
    if mask is not None:
        error = error * mask.float()
    return error
