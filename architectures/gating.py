import warnings

import torch
import numpy as np
from torch import Tensor
from typing import Tuple, Optional


class GatingNetwork:
    """
    Deterministic hard-routing gating for a 2D MoE grid over
    forward log moneyness (ln(F/K)) and maturity (T).

    No learned parameters. Routing is a quantile-boundary lookup.
    Training overlap is handled via Gaussian kernel weighting.
    """

    def __init__(
        self,
        n_moneyness_bins: int,
        n_maturity_bins: int,
        moneyness_quantiles: Tensor,
        maturity_quantiles: Tensor,
        kernel_bandwidth: float = 1.0,
        moneyness_col_idx: int = 0,
        maturity_col_idx: int = 3,
        data_moneyness_range: Optional[Tuple[float, float]] = None,
        data_maturity_range: Optional[Tuple[float, float]] = None,
    ):
        self.M = n_moneyness_bins
        self.N = n_maturity_bins
        self.kernel_bandwidth = kernel_bandwidth
        self.moneyness_col_idx = moneyness_col_idx
        self.maturity_col_idx = maturity_col_idx
        self.data_moneyness_range = data_moneyness_range
        self.data_maturity_range = data_maturity_range

        # --- validate shapes ---
        if moneyness_quantiles.shape != (self.M - 1,):
            raise ValueError(
                f"Expected moneyness_quantiles shape ({self.M - 1},), "
                f"got {moneyness_quantiles.shape}"
            )
        if maturity_quantiles.shape != (self.N - 1,):
            raise ValueError(
                f"Expected maturity_quantiles shape ({self.N - 1},), "
                f"got {maturity_quantiles.shape}"
            )

        # --- validate strictly increasing ---
        if self.M > 2:
            diffs = moneyness_quantiles[1:] - moneyness_quantiles[:-1]
            if (diffs <= 0).any():
                raise ValueError(
                    "moneyness_quantiles must be strictly increasing, "
                    f"got {moneyness_quantiles.tolist()}"
                )
        if self.N > 2:
            diffs = maturity_quantiles[1:] - maturity_quantiles[:-1]
            if (diffs <= 0).any():
                raise ValueError(
                    "maturity_quantiles must be strictly increasing, "
                    f"got {maturity_quantiles.tolist()}"
                )

        self.moneyness_quantiles = moneyness_quantiles.contiguous()
        self.maturity_quantiles = maturity_quantiles.contiguous()

        # --- precompute centers and scales ---
        self._moneyness_centers = self._compute_centers(
            self.moneyness_quantiles, self.M, data_moneyness_range
        )
        self._maturity_centers = self._compute_centers(
            self.maturity_quantiles, self.N, data_maturity_range
        )
        self._moneyness_scale = self._median_bin_width(self._moneyness_centers)
        self._maturity_scale = self._median_bin_width(self._maturity_centers)

        self._device = self.moneyness_quantiles.device

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------

    def to(self, device) -> "GatingNetwork":
        """Move all internal tensors to *device*. Returns self for chaining."""
        self.moneyness_quantiles = self.moneyness_quantiles.to(device)
        self.maturity_quantiles = self.maturity_quantiles.to(device)
        self._moneyness_centers = self._moneyness_centers.to(device)
        self._maturity_centers = self._maturity_centers.to(device)
        self._moneyness_scale = self._moneyness_scale.to(device)
        self._maturity_scale = self._maturity_scale.to(device)
        self._device = device
        return self

    def _check_device(self, x: Tensor) -> None:
        if x.device != self._device:
            raise RuntimeError(
                f"Input tensor is on {x.device} but GatingNetwork is on "
                f"{self._device}. Call gating.to({x.device}) first."
            )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_data(
        cls,
        x: Tensor,
        n_moneyness_bins: int,
        n_maturity_bins: int,
        strategy: str = "quantile",
        kernel_bandwidth: float = 1.0,
        moneyness_col_idx: int = 0,
        maturity_col_idx: int = 3,
        y: Optional[Tensor] = None,
    ) -> "GatingNetwork":
        """
        Compute bin boundaries from training data for regional specialization.

        Args:
            x: Training feature tensor, shape (N_samples, n_features).
            n_moneyness_bins: M.
            n_maturity_bins: N.
            strategy:
                "quantile"  — equal-count percentile bins.
                "kmeans"    — 1D k-means on each axis (natural regime breaks).
                "variance"  — DP-optimal boundaries minimizing within-bin
                              target variance. Requires *y*.
            kernel_bandwidth: Gaussian kernel bandwidth.
            moneyness_col_idx: Column index of fwd log moneyness in x.
            maturity_col_idx: Column index of T in x.
            y: Target tensor, shape (N_samples,) or (N_samples, d).
               Required for "variance".

        Returns:
            A fully constructed GatingNetwork.
        """
        m_vals = x[:, moneyness_col_idx].detach().cpu().numpy()
        t_vals = x[:, maturity_col_idx].detach().cpu().numpy()

        data_moneyness_range = (float(m_vals.min()), float(m_vals.max()))
        data_maturity_range = (float(t_vals.min()), float(t_vals.max()))

        if strategy == "quantile":
            mq = cls._quantile_boundaries(m_vals, n_moneyness_bins)
            tq = cls._quantile_boundaries(t_vals, n_maturity_bins)

        elif strategy == "kmeans":
            mq = cls._kmeans_boundaries(m_vals, n_moneyness_bins)
            tq = cls._kmeans_boundaries(t_vals, n_maturity_bins)

        elif strategy == "variance":
            if y is None:
                raise ValueError("strategy='variance' requires `y` tensor")
            y_np = y.detach().cpu().numpy()
            min_per_bin = max(
                50,
                len(m_vals) // (max(n_moneyness_bins, n_maturity_bins) * 10),
            )
            mq = cls._variance_optimal_boundaries(
                m_vals, y_np, n_moneyness_bins, min_samples_per_bin=min_per_bin
            )
            tq = cls._variance_optimal_boundaries(
                t_vals, y_np, n_maturity_bins, min_samples_per_bin=min_per_bin
            )

        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")

        # Deduplicate / enforce strict monotonicity
        mq = cls._enforce_strict_increasing(mq)
        tq = cls._enforce_strict_increasing(tq)

        return cls(
            n_moneyness_bins=n_moneyness_bins,
            n_maturity_bins=n_maturity_bins,
            moneyness_quantiles=torch.tensor(mq, dtype=x.dtype),
            maturity_quantiles=torch.tensor(tq, dtype=x.dtype),
            kernel_bandwidth=kernel_bandwidth,
            moneyness_col_idx=moneyness_col_idx,
            maturity_col_idx=maturity_col_idx,
            data_moneyness_range=data_moneyness_range,
            data_maturity_range=data_maturity_range,
        )

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def route(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Hard routing — assigns each sample to the cell whose center is
        nearest on each axis independently.  This guarantees that the
        routed cell is the one with the highest Gaussian training weight.

        Args:
            x: (batch_size, n_features) full feature tensor.

        Returns:
            moneyness_bins: (batch_size,) in [0, M-1]
            maturity_bins:  (batch_size,) in [0, N-1]
        """
        self._check_device(x)

        m_vals = x[:, self.moneyness_col_idx]  # (B,)
        t_vals = x[:, self.maturity_col_idx]   # (B,)

        # Nearest center on each axis — consistent with Gaussian weight peak
        moneyness_bins = (m_vals.unsqueeze(1) - self._moneyness_centers).abs().argmin(dim=1)
        maturity_bins = (t_vals.unsqueeze(1) - self._maturity_centers).abs().argmin(dim=1)

        return moneyness_bins, maturity_bins

    def get_training_weights(self, x: Tensor) -> Tensor:
        """
        Gaussian kernel weights for every sample relative to every cell center.

        Args:
            x: (batch_size, n_features)

        Returns:
            weights: (batch_size, M, N) — raw (unnormalized) Gaussian weights.
        """
        self._check_device(x)

        m_vals = x[:, self.moneyness_col_idx]  # (B,)
        t_vals = x[:, self.maturity_col_idx]   # (B,)

        # Normalized displacement per axis
        dm = (m_vals.unsqueeze(1) - self._moneyness_centers.unsqueeze(0)) / self._moneyness_scale  # (B, M)
        dt = (t_vals.unsqueeze(1) - self._maturity_centers.unsqueeze(0)) / self._maturity_scale    # (B, N)

        # Squared distances on the 2D grid: (B, M, N)
        dist_sq = dm.unsqueeze(2) ** 2 + dt.unsqueeze(1) ** 2

        weights = torch.exp(-0.5 * dist_sq / (self.kernel_bandwidth ** 2))
        return weights

    def get_cell_centers(self) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            moneyness_centers: (M,)
            maturity_centers:  (N,)
        """
        return self._moneyness_centers.clone(), self._maturity_centers.clone()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_bin_counts(self, x: Tensor) -> Tensor:
        """
        Count samples per cell under hard routing.

        Returns:
            counts: (M, N)
        """
        m_bins, t_bins = self.route(x)
        flat = m_bins * self.N + t_bins
        counts = torch.bincount(flat, minlength=self.M * self.N)
        return counts.reshape(self.M, self.N)

    def get_expert_index(self, moneyness_bin: int, maturity_bin: int) -> int:
        """2D grid coords -> flat index (row-major)."""
        return moneyness_bin * self.N + maturity_bin

    def describe_cell(self, i: int, j: int) -> dict:
        """Human-readable info for grid cell (i, j)."""
        mq = self.moneyness_quantiles
        tq = self.maturity_quantiles

        m_lo = float(mq[i - 1]) if i > 0 else None
        m_hi = float(mq[i]) if i < self.M - 1 else None
        t_lo = float(tq[j - 1]) if j > 0 else None
        t_hi = float(tq[j]) if j < self.N - 1 else None

        mc = float(self._moneyness_centers[i])
        tc = float(self._maturity_centers[j])

        m_label = self._regime_label_moneyness(i, self.M)
        t_label = self._regime_label_maturity(j, self.N)

        return {
            "moneyness_range": (m_lo, m_hi),
            "maturity_range": (t_lo, t_hi),
            "center": (mc, tc),
            "flat_index": self.get_expert_index(i, j),
            "regime": f"{m_label}, {t_label}",
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_centers(
        quantiles: Tensor,
        n_bins: int,
        data_range: Optional[Tuple[float, float]] = None,
    ) -> Tensor:
        """
        Compute bin centers from quantile boundaries.

        Interior bins: midpoint of adjacent boundaries.
        Edge bins: midpoint between boundary and data_range endpoint (if
        provided), else mirror the nearest interior half-width outward.
        """
        if n_bins == 1:
            if data_range is not None:
                return torch.tensor(
                    [(data_range[0] + data_range[1]) / 2.0],
                    dtype=quantiles.dtype,
                )
            return torch.tensor([0.0], dtype=quantiles.dtype)

        centers = torch.empty(n_bins, dtype=quantiles.dtype)

        # Interior bins (indices 1 .. K-2)
        for k in range(1, n_bins - 1):
            centers[k] = (quantiles[k - 1] + quantiles[k]) / 2.0

        # Left edge bin (index 0)
        if data_range is not None:
            centers[0] = (data_range[0] + float(quantiles[0])) / 2.0
        elif n_bins >= 3:
            centers[0] = float(quantiles[0]) - (float(quantiles[1]) - float(quantiles[0])) / 2.0
        else:  # n_bins == 2
            centers[0] = float(quantiles[0]) - max(abs(float(quantiles[0])), 0.1) * 0.5

        # Right edge bin (index K-1)
        if data_range is not None:
            centers[-1] = (float(quantiles[-1]) + data_range[1]) / 2.0
        elif n_bins >= 3:
            centers[-1] = float(quantiles[-1]) + (float(quantiles[-1]) - float(quantiles[-2])) / 2.0
        else:  # n_bins == 2
            centers[-1] = float(quantiles[0]) + max(abs(float(quantiles[0])), 0.1) * 0.5

        return centers

    @staticmethod
    def _median_bin_width(centers: Tensor) -> Tensor:
        """Median spacing between adjacent bin centers. Clamped to >= 1e-8."""
        if centers.numel() <= 1:
            return torch.tensor(1.0, dtype=centers.dtype)
        diffs = centers[1:] - centers[:-1]
        return diffs.median().clamp(min=1e-8)

    @staticmethod
    def _regime_label_moneyness(i: int, M: int) -> str:
        """Derive moneyness regime label from relative grid position."""
        labels = ["deep OTM", "OTM", "ATM", "ITM", "deep ITM"]
        if M >= 5:
            frac = i / (M - 1)
            idx = min(int(frac * 5), 4)
            return labels[idx]
        # Fewer bins — pick evenly spaced subset
        if M == 1:
            return "ATM"
        if M == 2:
            return ["OTM", "ITM"][i]
        if M == 3:
            return ["OTM", "ATM", "ITM"][i]
        # M == 4
        return ["deep OTM", "OTM", "ITM", "deep ITM"][i]

    @staticmethod
    def _regime_label_maturity(j: int, N: int) -> str:
        """Derive maturity regime label from relative grid position."""
        labels = ["near-expiry", "short-dated", "medium-dated", "long-dated"]
        if N >= 4:
            frac = j / (N - 1)
            idx = min(int(frac * 4), 3)
            return labels[idx]
        if N == 1:
            return "medium-dated"
        if N == 2:
            return ["short-dated", "long-dated"][j]
        # N == 3
        return ["near-expiry", "medium-dated", "long-dated"][j]

    @staticmethod
    def _enforce_strict_increasing(boundaries: np.ndarray) -> np.ndarray:
        """
        Ensure boundaries are strictly increasing. Nudge duplicates apart
        by 1e-10 increments. Warn if nudging was needed.
        """
        if len(boundaries) <= 1:
            return boundaries
        nudged = False
        for i in range(1, len(boundaries)):
            if boundaries[i] <= boundaries[i - 1]:
                boundaries[i] = boundaries[i - 1] + 1e-10
                nudged = True
        if nudged:
            warnings.warn(
                "Duplicate boundaries detected and nudged apart by 1e-10. "
                "This can happen with repeated axis values in the data.",
                stacklevel=3,
            )
        return boundaries

    # ------------------------------------------------------------------
    # Boundary computation strategies
    # ------------------------------------------------------------------

    @staticmethod
    def _quantile_boundaries(vals: np.ndarray, n_bins: int) -> np.ndarray:
        """Equal-count percentile boundaries. Returns (n_bins - 1,) array."""
        percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]
        return np.percentile(vals, percentiles)

    @staticmethod
    def _kmeans_boundaries(vals: np.ndarray, n_bins: int) -> np.ndarray:
        """
        1D k-means clustering to find natural breaks.
        Returns sorted (n_bins - 1,) boundary array (midpoints between
        adjacent cluster centers).
        """
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            raise ImportError(
                "strategy='kmeans' requires scikit-learn. "
                "Install it with: pip install scikit-learn"
            )

        km = KMeans(n_clusters=n_bins, n_init=10, random_state=42)
        km.fit(vals.reshape(-1, 1))
        centers = np.sort(km.cluster_centers_.ravel())
        boundaries = (centers[:-1] + centers[1:]) / 2.0
        return boundaries

    @staticmethod
    def _variance_optimal_boundaries(
        vals: np.ndarray,
        targets: np.ndarray,
        n_bins: int,
        n_candidates: int = 500,
        min_samples_per_bin: int = 50,
    ) -> np.ndarray:
        """
        Find boundary placements minimizing total within-bin target SSD
        via dynamic programming on a discretized axis.

        Segments with fewer than *min_samples_per_bin* points get infinite
        cost to prevent degenerate bins.

        After backtracking, duplicate boundaries are nudged apart.
        """
        if targets.ndim == 1:
            targets = targets[:, None]

        n = len(vals)
        order = np.argsort(vals)
        sorted_vals = vals[order]
        sorted_y = targets[order]
        d = sorted_y.shape[1]

        Q = min(n_candidates, n)
        fine_edges = np.linspace(0, n, Q + 1, dtype=int)

        # Precompute per-fine-bin aggregates
        counts = np.zeros(Q)
        sums = np.zeros((Q, d))
        sum_sqs = np.zeros((Q, d))
        for q in range(Q):
            a, b = fine_edges[q], fine_edges[q + 1]
            block = sorted_y[a:b]
            counts[q] = b - a
            sums[q] = block.sum(axis=0)
            sum_sqs[q] = (block ** 2).sum(axis=0)

        pcounts = np.concatenate([[0], np.cumsum(counts)])
        psums = np.vstack([[np.zeros(d)], np.cumsum(sums, axis=0)])
        psum_sqs = np.vstack([[np.zeros(d)], np.cumsum(sum_sqs, axis=0)])

        def segment_ssd(a: int, b: int) -> float:
            c = pcounts[b] - pcounts[a]
            if c < min_samples_per_bin:
                return float("inf")
            s = psums[b] - psums[a]
            ss = psum_sqs[b] - psum_sqs[a]
            return float((ss - s ** 2 / c).sum())

        INF = float("inf")
        dp = np.full((n_bins + 1, Q + 1), INF)
        split = np.zeros((n_bins + 1, Q + 1), dtype=int)
        dp[0][0] = 0.0

        for q in range(1, Q + 1):
            dp[1][q] = segment_ssd(0, q)

        for k in range(2, n_bins + 1):
            for q in range(k, Q + 1):
                best = INF
                best_j = k - 1
                for j in range(k - 1, q):
                    cost = dp[k - 1][j] + segment_ssd(j, q)
                    if cost < best:
                        best = cost
                        best_j = j
                dp[k][q] = best
                split[k][q] = best_j

        # Backtrack
        splits = []
        q = Q
        for k in range(n_bins, 1, -1):
            q = split[k][q]
            splits.append(q)
        splits.reverse()

        # Convert fine-bin indices to axis values
        boundaries = np.empty(len(splits))
        for i, s in enumerate(splits):
            idx_left = fine_edges[s] - 1
            idx_right = fine_edges[s]
            boundaries[i] = (sorted_vals[idx_left] + sorted_vals[idx_right]) / 2.0

        return boundaries
