import warnings
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn


# Supported activation constructors keyed by name
_ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "silu": nn.SiLU,
}


class ExpertPool(nn.Module):
    """
    Collection of M×N independent feedforward expert networks for
    hard-gated Mixture-of-Experts option pricing.

    Each expert maps (input_dim,) → (output_dim,).  Experts are indexed
    row-major: flat_idx = i * N + j, consistent with
    GatingNetwork.get_expert_index().

    No dropout, no batch norm, no output activation — targets are
    z-scored and unbounded.
    """

    def __init__(
        self,
        n_moneyness_bins: int,
        n_maturity_bins: int,
        input_dim: int = 23,
        output_dim: int = 1,
        default_hidden_dims: Optional[List[int]] = None,
        default_activation: str = "relu",
        batch_norm: bool = False,
    ):
        super().__init__()

        if default_hidden_dims is None:
            raise ValueError(
                "default_hidden_dims must be provided (e.g. [64, 32] from "
                "elbow analysis). Refusing to silently hardcode an architecture."
            )

        if default_activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation {default_activation!r}. "
                f"Supported: {list(_ACTIVATIONS.keys())}"
            )

        self.M = n_moneyness_bins
        self.N = n_maturity_bins
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.default_hidden_dims = list(default_hidden_dims)
        self.default_activation = default_activation
        self.batch_norm = batch_norm

        # Per-expert metadata (hidden_dims, activation) for diagnostics
        n_experts = self.M * self.N
        self._expert_hidden_dims: List[List[int]] = [
            list(default_hidden_dims) for _ in range(n_experts)
        ]
        self._expert_activations: List[str] = [
            default_activation for _ in range(n_experts)
        ]

        self.experts = nn.ModuleList(
            [
                self._build_expert(
                    input_dim, default_hidden_dims, output_dim, default_activation,
                    batch_norm=batch_norm,
                )
                for _ in range(n_experts)
            ]
        )

    # ------------------------------------------------------------------
    # Expert construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_expert(
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str,
        batch_norm: bool = False,
    ) -> nn.Sequential:
        """
        Build a feedforward expert: Linear → [BN →] Act → … → Linear (no output activation).
        """
        act_cls = _ACTIVATIONS[activation]
        layers: list = []

        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act_cls())
            prev_dim = h

        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Forward methods
    # ------------------------------------------------------------------

    def forward_expert(self, expert_idx: int, x: torch.Tensor) -> torch.Tensor:
        """
        Run a single expert on input data.

        Args:
            expert_idx: flat index in [0, M*N).
            x: (batch_size, input_dim)

        Returns:
            (batch_size, output_dim)
        """
        return self.experts[expert_idx](x)

    def forward_routed(
        self,
        x: torch.Tensor,
        moneyness_bins: torch.Tensor,
        maturity_bins: torch.Tensor,
    ) -> torch.Tensor:
        """
        Batch inference — each sample goes to its assigned expert.

        Uses torch.unique + boolean masking so only experts with routed
        samples are evaluated and no per-sample Python loops occur.

        Args:
            x: (batch_size, input_dim)
            moneyness_bins: (batch_size,) in [0, M-1]
            maturity_bins:  (batch_size,) in [0, N-1]

        Returns:
            (batch_size, output_dim)
        """
        flat_indices = moneyness_bins * self.N + maturity_bins
        out = torch.zeros(
            x.shape[0], self.output_dim, device=x.device, dtype=x.dtype
        )

        for idx in torch.unique(flat_indices):
            mask = flat_indices == idx
            out[mask] = self.experts[idx.item()](x[mask])

        return out

    def forward_all(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run every expert on all inputs.

        Args:
            x: (batch_size, input_dim)

        Returns:
            (batch_size, M*N)  — column k is expert k's prediction.
        """
        preds = [expert(x) for expert in self.experts]  # list of (B, 1)
        return torch.cat(preds, dim=-1)  # (B, M*N)

    # ------------------------------------------------------------------
    # Expert replacement
    # ------------------------------------------------------------------

    def replace_expert(
        self,
        i: int,
        j: int,
        new_hidden_dims: List[int],
        new_activation: Optional[str] = None,
    ) -> None:
        """
        Replace expert at grid position (i, j) with a freshly initialised
        network of (possibly different) architecture.  Previous weights
        are discarded.

        Args:
            i: moneyness bin index.
            j: maturity bin index.
            new_hidden_dims: hidden layer widths for the replacement.
            new_activation: activation name; defaults to self.default_activation.
        """
        act = new_activation if new_activation is not None else self.default_activation
        if act not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation {act!r}. Supported: {list(_ACTIVATIONS.keys())}"
            )

        flat_idx = i * self.N + j

        # Determine device from existing experts
        device = next(self.experts[flat_idx].parameters()).device

        new_expert = self._build_expert(
            self.input_dim, new_hidden_dims, self.output_dim, act,
            batch_norm=self.batch_norm,
        ).to(device)

        self.experts[flat_idx] = new_expert
        self._expert_hidden_dims[flat_idx] = list(new_hidden_dims)
        self._expert_activations[flat_idx] = act

        warnings.warn(
            f"Expert ({i}, {j}) [flat {flat_idx}] replaced with "
            f"hidden_dims={new_hidden_dims}, activation={act!r}. "
            f"This expert needs retraining.",
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------

    def create_optimizers(
        self, lr: float = 1e-3, weight_decay: float = 0.0
    ) -> List[torch.optim.Adam]:
        """
        One Adam optimizer per expert — experts train independently with
        their own kernel-weighted losses.

        Returns:
            List of M*N Adam optimizers.
        """
        return [
            torch.optim.Adam(expert.parameters(), lr=lr, weight_decay=weight_decay)
            for expert in self.experts
        ]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_expert_info(self, i: int, j: int) -> Dict[str, Any]:
        """Human-readable info for expert at grid position (i, j)."""
        flat_idx = i * self.N + j
        n_params = sum(p.numel() for p in self.experts[flat_idx].parameters())
        return {
            "grid_position": (i, j),
            "flat_index": flat_idx,
            "n_parameters": n_params,
            "hidden_dims": self._expert_hidden_dims[flat_idx],
            "activation": self._expert_activations[flat_idx],
        }

    def get_total_parameters(self) -> int:
        """Sum of parameters across all experts."""
        return sum(p.numel() for p in self.parameters())

    def get_parameter_distribution(self) -> Dict[str, Any]:
        """Parameter count statistics across experts."""
        counts = [
            sum(p.numel() for p in expert.parameters())
            for expert in self.experts
        ]
        t = torch.tensor(counts, dtype=torch.float)
        return {
            "total": int(t.sum().item()),
            "per_expert_mean": float(t.mean().item()),
            "per_expert_std": float(t.std().item()),
            "per_expert_counts": counts,
        }
