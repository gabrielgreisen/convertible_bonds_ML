import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Dict, Any


class MoEPricer(nn.Module):
    """
    Top-level orchestrator composing a GatingNetwork and ExpertPool
    for hard-gated Mixture-of-Experts option pricing.

    The GatingNetwork handles deterministic routing (no learned params).
    The ExpertPool holds M×N specialist MLPs, one per grid cell.

    Usage:
        gating = GatingNetwork.from_data(x_train, M, N)
        pool = ExpertPool(M, N, input_dim=23)
        pricer = MoEPricer(gating, pool)
        optimizers = pool.create_optimizers(lr=1e-3)
        for epoch in range(n_epochs):
            loss = pricer.train_epoch(train_loader, optimizers, device)
        metrics = pricer.evaluate(test_loader, device)
    """

    def __init__(self, gating, expert_pool: nn.Module):
        super().__init__()
        self.gating = gating  # not an nn.Module — stored as plain attribute
        self.expert_pool = expert_pool  # nn.Module — registered as submodule

        # Move gating tensors to the same device as the expert pool
        device = next(expert_pool.parameters()).device
        self.gating.to(device)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def forward(self, x: Tensor, x_raw: Tensor = None) -> Tensor:
        """
        Hard-routed inference: one gating lookup, one expert forward per sample.

        Args:
            x: (batch_size, n_features) scaled feature tensor for experts.
            x_raw: (batch_size, n_features) optional raw (unscaled) features
                   for gating. If None, gating uses x directly.

        Returns:
            predictions: (batch_size, 1)
        """
        x_gate = x_raw if x_raw is not None else x
        m_bins, t_bins = self.gating.route(x_gate)
        return self.expert_pool.forward_routed(x, m_bins, t_bins)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_epoch(
        self,
        dataloader,
        optimizers: List,
        device,
        raw_features: bool = False,
    ) -> float:
        """
        Run one training epoch with kernel-weighted losses.

        Each expert is trained independently: zero_grad → forward → weighted
        MSE → backward → step.  Gradients are NOT accumulated across experts.

        Args:
            dataloader: yields (x, y) or (x_scaled, x_raw, y) batches.
            optimizers: list of M*N optimizers, one per expert (flat row-major).
            device: torch device.
            raw_features: if True, dataloader yields (x_scaled, x_raw, y)
                and gating routes on x_raw while experts receive x_scaled.

        Returns:
            Mean loss across all experts and batches.
        """
        self.expert_pool.train()
        M, N = self.gating.M, self.gating.N
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            if raw_features:
                x, x_raw, y = batch
                x, x_raw, y = x.to(device), x_raw.to(device), y.to(device)
                x_gate = x_raw
            else:
                x, y = batch
                x, y = x.to(device), y.to(device)
                x_gate = x

            y_flat = y.view(-1)  # ensure (B,)

            weights = self.gating.get_training_weights(x_gate)  # (B, M, N)

            batch_loss = 0.0
            for i in range(M):
                for j in range(N):
                    flat_idx = i * N + j
                    opt = optimizers[flat_idx]
                    expert = self.expert_pool.experts[flat_idx]

                    opt.zero_grad()
                    pred = expert(x)  # (B, 1)
                    w = weights[:, i, j]  # (B,)
                    residuals = pred.squeeze(-1) - y_flat  # (B,)
                    loss = (w * residuals.pow(2)).sum() / w.sum()
                    loss.backward()
                    opt.step()

                    batch_loss += loss.item()

            total_loss += batch_loss / (M * N)
            n_batches += 1

        return total_loss / max(n_batches, 1)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, dataloader, device, raw_features: bool = False) -> Dict[str, Any]:
        """
        Evaluate using hard routing (not kernel-weighted).

        Args:
            dataloader: yields (x, y) or (x_scaled, x_raw, y) batches.
            device: torch device.
            raw_features: if True, dataloader yields (x_scaled, x_raw, y).

        Returns:
            dict with:
                'mse': overall MSE (float)
                'per_cell_mse': (M, N) tensor of per-cell MSE
        """
        self.expert_pool.eval()
        M, N = self.gating.M, self.gating.N

        all_preds = []
        all_targets = []
        all_m_bins = []
        all_t_bins = []

        with torch.no_grad():
            for batch in dataloader:
                if raw_features:
                    x, x_raw, y = batch
                    x, x_raw, y = x.to(device), x_raw.to(device), y.to(device)
                    x_gate = x_raw
                else:
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                    x_gate = x

                pred = self.forward(x, x_gate if raw_features else None)
                m_bins, t_bins = self.gating.route(x_gate)

                all_preds.append(pred.squeeze(-1))  # (B,)
                all_targets.append(y.view(-1))       # (B,)
                all_m_bins.append(m_bins)
                all_t_bins.append(t_bins)

        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        m_bins = torch.cat(all_m_bins, dim=0)
        t_bins = torch.cat(all_t_bins, dim=0)

        overall_mse = (preds - targets).pow(2).mean().item()

        per_cell_mse = torch.zeros(M, N)
        for i in range(M):
            for j in range(N):
                mask = (m_bins == i) & (t_bins == j)
                if mask.any():
                    per_cell_mse[i, j] = (
                        (preds[mask] - targets[mask]).pow(2).mean().item()
                    )

        return {"mse": overall_mse, "per_cell_mse": per_cell_mse}

    def evaluate_per_cell(self, dataloader, device, raw_features: bool = False) -> List[Dict[str, Any]]:
        """
        Detailed per-cell evaluation for post-training analysis.

        Args:
            dataloader: yields (x, y) or (x_scaled, x_raw, y) batches.
            device: torch device.
            raw_features: if True, dataloader yields (x_scaled, x_raw, y).

        Returns:
            List of dicts, one per cell, each containing:
                'cell': (i, j)
                'sample_count': int
                'mse': float
                'mae': float
                'max_abs_error': float
                'regime': dict from gating.describe_cell(i, j)
        """
        self.expert_pool.eval()
        M, N = self.gating.M, self.gating.N

        all_preds = []
        all_targets = []
        all_m_bins = []
        all_t_bins = []

        with torch.no_grad():
            for batch in dataloader:
                if raw_features:
                    x, x_raw, y = batch
                    x, x_raw, y = x.to(device), x_raw.to(device), y.to(device)
                    x_gate = x_raw
                else:
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                    x_gate = x

                pred = self.forward(x, x_gate if raw_features else None)
                m_bins, t_bins = self.gating.route(x_gate)

                all_preds.append(pred.squeeze(-1))
                all_targets.append(y.view(-1))
                all_m_bins.append(m_bins)
                all_t_bins.append(t_bins)

        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        m_bins = torch.cat(all_m_bins, dim=0)
        t_bins = torch.cat(all_t_bins, dim=0)

        results = []
        for i in range(M):
            for j in range(N):
                mask = (m_bins == i) & (t_bins == j)
                count = mask.sum().item()

                if count > 0:
                    errors = preds[mask] - targets[mask]
                    abs_errors = errors.abs()
                    mse = errors.pow(2).mean().item()
                    mae = abs_errors.mean().item()
                    max_abs = abs_errors.max().item()
                else:
                    mse = 0.0
                    mae = 0.0
                    max_abs = 0.0

                results.append(
                    {
                        "cell": (i, j),
                        "sample_count": count,
                        "mse": mse,
                        "mae": mae,
                        "max_abs_error": max_abs,
                        "regime": self.gating.describe_cell(i, j),
                    }
                )

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save the full MoEPricer to disk.

        Saves:
            - GatingNetwork via pickle (no learned params)
            - ExpertPool state_dict (learned weights)
            - ExpertPool architecture via pickle (for reconstruction)
            - Grid dimensions M, N
        """
        torch.save(
            {
                "gating": self.gating,
                "expert_pool": self.expert_pool,
                "expert_pool_state_dict": self.expert_pool.state_dict(),
                "M": self.gating.M,
                "N": self.gating.N,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device=None) -> "MoEPricer":
        """
        Reconstruct a full MoEPricer from a saved file.

        Args:
            path: file path saved by MoEPricer.save().
            device: optional device to map tensors to.

        Returns:
            Reconstructed MoEPricer with identical predictions.
        """
        map_location = device if device is not None else "cpu"
        data = torch.load(path, map_location=map_location, weights_only=False)

        gating = data["gating"]
        expert_pool = data["expert_pool"]
        expert_pool.load_state_dict(data["expert_pool_state_dict"])

        return cls(gating, expert_pool)
