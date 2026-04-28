"""
Temporal Graph Neural Network (T-GNN) for PV power prediction and anomaly detection.

Architecture from: Mukherjee et al., "Temporal Graph Neural Networks for Early
Anomaly Detection and Performance Prediction via PV System Monitoring Data",
EUPVSEC 2025 (arXiv:2512.03114v1).

The model integrates:
  1. Graph Convolutional Network (GCN) \u2014 spatial dependencies between PV parameters
  2. Gated Recurrent Unit (GRU)        \u2014 temporal evolution across time steps
  3. Fully Connected output layer      \u2014 maps hidden state to power prediction

Adapted to the Costa PV Fault Dataset:
  Input nodes : irr, pvt, vdc1, vdc2, idc1, idc2  (6 parameters)
  Target      : pdc (total DC power)

No PyTorch Geometric dependency \u2014 GCN is implemented via adjacency matrix ops.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# ============================================================================
# GRAPH CONVOLUTION LAYER (manual, no PyG)
# ============================================================================


class GCNConv(nn.Module):
    """
    Single-layer Graph Convolution following Kipf & Welling (2017).

    Computes: H' = \u03c3(D\u0303^{-1/2} A\u0303 D\u0303^{-1/2} H W)

    where A\u0303 = A + I  (adjacency with self-loops),
          D\u0303 = degree matrix of A\u0303,
          W  = learnable weight matrix.

    Parameters
    ----------
    in_features : int
        Dimension of each node's input feature vector.
    out_features : int
        Dimension of each node's output feature vector.
    bias : bool
        If True, add a learnable bias vector.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (batch, num_nodes, in_features)
            Node feature matrix.
        adj_norm : Tensor, shape (num_nodes, num_nodes)
            Pre-computed normalized adjacency: D\u0303^{-1/2} A\u0303 D\u0303^{-1/2}.

        Returns
        -------
        Tensor, shape (batch, num_nodes, out_features)
        """
        # x @ W  \u2192  (batch, num_nodes, out_features)
        support = torch.matmul(x, self.weight)
        # adj_norm @ support  \u2192  neighbourhood aggregation
        out = torch.matmul(adj_norm, support)
        if self.bias is not None:
            out = out + self.bias
        return out


# ============================================================================
# TEMPORAL GNN MODEL
# ============================================================================


class TemporalGNN(nn.Module):
    """
    Temporal Graph Neural Network (T-GNN).

    For each time step t in a sequence:
      1. GCN aggregates spatial information across graph nodes.
      2. GRU updates the temporal hidden state per node.
    After processing the full sequence, a FC head on the aggregated hidden
    state predicts the target (power output).

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes (PV parameters).
    node_feature_dim : int
        Dimensionality of each node's raw feature (typically 1 for scalar sensor).
    gcn_hidden_dim : int
        Output dimension of the GCN layer.
    gru_hidden_dim : int
        Hidden dimension of the GRU cell.
    output_dim : int
        Dimension of the final output (1 for scalar power prediction).
    dropout : float
        Dropout rate applied after GCN and before the output head.
    """

    def __init__(
        self,
        num_nodes: int = 6,
        node_feature_dim: int = 1,
        gcn_hidden_dim: int = 32,
        gru_hidden_dim: int = 64,
        output_dim: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_feature_dim = node_feature_dim
        self.gcn_hidden_dim = gcn_hidden_dim
        self.gru_hidden_dim = gru_hidden_dim

        # Spatial: Graph Convolution
        self.gcn = GCNConv(node_feature_dim, gcn_hidden_dim)

        # Temporal: Gated Recurrent Unit
        # GRU input = gcn_hidden_dim (per node), hidden = gru_hidden_dim
        self.gru_cell = nn.GRUCell(
            input_size=gcn_hidden_dim,
            hidden_size=gru_hidden_dim,
        )

        self.dropout = nn.Dropout(dropout)

        # Output head: maps from gru_hidden_dim \u2192 output_dim
        # We aggregate across nodes (mean pool) then project.
        self.fc_out = nn.Sequential(
            nn.Linear(gru_hidden_dim, gru_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden_dim // 2, output_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        adj_norm: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (batch, seq_len, num_nodes, node_feature_dim)
            Time series of node features.
        adj_norm : Tensor, shape (num_nodes, num_nodes)
            Normalized adjacency matrix (shared across batch and time).

        Returns
        -------
        Tensor, shape (batch, output_dim)
            Predicted target value.
        """
        batch_size, seq_len, num_nodes, feat_dim = x.shape

        # Initialize GRU hidden state: (batch * num_nodes, gru_hidden_dim)
        h = torch.zeros(
            batch_size * num_nodes,
            self.gru_hidden_dim,
            device=x.device,
            dtype=x.dtype,
        )

        for t in range(seq_len):
            # x_t: (batch, num_nodes, feat_dim)
            x_t = x[:, t, :, :]

            # GCN: spatial aggregation \u2192 (batch, num_nodes, gcn_hidden_dim)
            gcn_out = F.relu(self.gcn(x_t, adj_norm))
            gcn_out = self.dropout(gcn_out)

            # Reshape for GRU: (batch * num_nodes, gcn_hidden_dim)
            gcn_flat = gcn_out.reshape(batch_size * num_nodes, self.gcn_hidden_dim)

            # GRU: temporal update \u2192 (batch * num_nodes, gru_hidden_dim)
            h = self.gru_cell(gcn_flat, h)

        # h: (batch * num_nodes, gru_hidden_dim) \u2192 (batch, num_nodes, gru_hidden_dim)
        h = h.reshape(batch_size, num_nodes, self.gru_hidden_dim)

        # Aggregate across nodes: mean pooling \u2192 (batch, gru_hidden_dim)
        h_pooled = h.mean(dim=1)

        # Output head \u2192 (batch, output_dim)
        return self.fc_out(h_pooled)


# ============================================================================
# ADJACENCY MATRIX UTILITIES
# ============================================================================


def build_adjacency_matrix(num_nodes: int, directed: bool = True) -> torch.Tensor:
    """
    Build a fully-connected adjacency matrix (with self-loops).

    For the T-GNN, all PV parameters are considered inter-dependent.
    The paper uses directed causal edges; we use a fully-connected graph
    and let the GCN weights learn the relevant spatial relationships.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    directed : bool
        If True, the adjacency is asymmetric (directed). If False, symmetric.

    Returns
    -------
    Tensor, shape (num_nodes, num_nodes)
        Normalized adjacency matrix: D\u0303^{-1/2} A\u0303 D\u0303^{-1/2}.
    """
    # Fully connected + self-loops
    A = torch.ones(num_nodes, num_nodes)

    # Normalize: D\u0303^{-1/2} A\u0303 D\u0303^{-1/2}
    D = A.sum(dim=1)  # degree vector
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D))
    adj_norm = D_inv_sqrt @ A @ D_inv_sqrt

    return adj_norm


def build_causal_adjacency(
    node_names: list[str],
    causal_edges: list[tuple[str, str]] | None = None,
) -> torch.Tensor:
    """
    Build a directed causal adjacency matrix from named edges.

    If causal_edges is None, uses domain-knowledge defaults for Costa:
      irr \u2192 pvt, irr \u2192 vdc1, irr \u2192 vdc2, irr \u2192 idc1, irr \u2192 idc2
      pvt \u2192 vdc1, pvt \u2192 vdc2
      vdc1 \u2192 idc1, vdc2 \u2192 idc2

    Self-loops are always added.

    Returns the normalized adjacency matrix.
    """
    n = len(node_names)
    name_to_idx = {name: i for i, name in enumerate(node_names)}

    if causal_edges is None:
        causal_edges = [
            ("irr", "pvt"),
            ("irr", "vdc1"),
            ("irr", "vdc2"),
            ("irr", "idc1"),
            ("irr", "idc2"),
            ("pvt", "vdc1"),
            ("pvt", "vdc2"),
            ("vdc1", "idc1"),
            ("vdc2", "idc2"),
        ]

    # Start with self-loops
    A = torch.eye(n)
    for src, dst in causal_edges:
        if src in name_to_idx and dst in name_to_idx:
            A[name_to_idx[src], name_to_idx[dst]] = 1.0

    # Normalize
    D = A.sum(dim=1).clamp(min=1e-12)
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D))
    adj_norm = D_inv_sqrt @ A @ D_inv_sqrt

    return adj_norm


# ============================================================================
# DATASET
# ============================================================================

# Default Costa columns used as graph nodes
COSTA_INPUT_NODES: list[str] = ["irr", "pvt", "vdc1", "vdc2", "idc1", "idc2"]
COSTA_TARGET_COL: str = "pdc"


class CostaGraphDataset(Dataset):
    """
    PyTorch Dataset for the Costa PV Fault Dataset, structured for T-GNN.

    Each sample is a sliding window of `seq_len` consecutive time steps.
    Node features at each time step are the (normalized) sensor readings.
    The target is the normalized total DC power at the last time step.

    Parameters
    ----------
    data : np.ndarray, shape (N, num_features)
        Feature matrix (input columns stacked).
    targets : np.ndarray, shape (N,)
        Target values.
    seq_len : int
        Length of each temporal window.
    stride : int
        Step size between consecutive windows.
    """

    def __init__(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        seq_len: int = 10,
        stride: int = 1,
    ):
        super().__init__()
        self.data = data.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.seq_len = seq_len
        self.stride = stride
        self.num_nodes = data.shape[1]

        # Pre-compute valid window start indices
        self.indices = list(range(0, len(data) - seq_len, stride))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.indices[idx]
        end = start + self.seq_len

        # x: (seq_len, num_nodes, 1

        # Each node has a single scalar feature at each time step
        x = self.data[start:end]  # (seq_len, num_nodes)
        x = x[:, :, np.newaxis]   # (seq_len, num_nodes, 1)

        # Target: power at the last time step of the window
        y = self.targets[end - 1]  # scalar

        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


# ============================================================================
# DATA LOADING
# ============================================================================


def load_costa_for_tgnn(
    parquet_path: str | Path | None = None,
    input_nodes: list[str] | None = None,
    target_col: str = COSTA_TARGET_COL,
    daytime_irr_threshold: float = 50.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Load Costa parquet and prepare arrays for T-GNN training.

    Applies MinMax normalization (per paper).

    Parameters
    ----------
    parquet_path : str or Path, optional
        Path to the ingested Costa parquet. If None, uses default location.
    input_nodes : list[str], optional
        Columns to use as graph nodes. Defaults to COSTA_INPUT_NODES.
    target_col : str
        Column name for the prediction target.
    daytime_irr_threshold : float
        Minimum irradiance to keep (filters nighttime rows).

    Returns
    -------
    data : np.ndarray, shape (N, num_nodes)
        Normalized input features.
    targets : np.ndarray, shape (N,)
        Normalized target values.
    metadata : dict
        Contains scaler parameters, column info, label array for anomaly detection.
    """
    import pandas as pd

    if input_nodes is None:
        input_nodes = COSTA_INPUT_NODES.copy()

    if parquet_path is None:
        project_root = Path(__file__).resolve().parents[4]
        # Build path to data/interim/ingestion/costa/costa_merged.parquet based on project root
        parquet_path = project_root / "data" / "interim" / "ingestion" / "costa" / "costa_merged.parquet"

    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Costa parquet not found: {parquet_path}\n"
            "Run `python -m src.data.ingestion --dataset costa` first."
        )

    df = pd.read_parquet(parquet_path)

    # Filter daytime
    if "irr" in df.columns and daytime_irr_threshold > 0:
        df = df[df["irr"] >= daytime_irr_threshold].reset_index(drop=True)

    # Ensure required columns exist
    all_cols = input_nodes + [target_col]
    missing = [c for c in all_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in Costa parquet: {missing}")

    # Extract arrays
    data_raw = df[input_nodes].values.astype(np.float64)
    target_raw = df[target_col].values.astype(np.float64)

    # MinMax normalization (per paper)
    data_min = data_raw.min(axis=0)
    data_max = data_raw.max(axis=0)
    data_range = data_max - data_min
    data_range[data_range == 0] = 1.0  # prevent division by zero

    target_min = target_raw.min()
    target_max = target_raw.max()
    target_range = target_max - target_min
    if target_range == 0:
        target_range = 1.0

    data_norm = (data_raw - data_min) / data_range
    target_norm = (target_raw - target_min) / target_range

    # Labels for anomaly detection (if available)
    labels = df["label"].values if "label" in df.columns else None

    metadata = {
        "input_nodes": input_nodes,
        "target_col": target_col,
        "data_min": data_min,
        "data_max": data_max,
        "data_range": data_range,
        "target_min": target_min,
        "target_max": target_max,
        "target_range": target_range,
        "n_samples": len(data_norm),
        "labels": labels,
    }

    return data_norm.astype(np.float32), target_norm.astype(np.float32), metadata
