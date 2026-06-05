# nnue/feature_transformer.py

from __future__ import annotations

from typing import Iterable, List
import torch
import torch.nn as nn

from .config import CONFIG


class FeatureTransformer(nn.Module):
    """
    NNUE Feature Transformer Layer (L1).

    Converts sparse active coordinate feature indices into a dense, continuous
    vector accumulator state representation.

    Mathematical Representation:
        Accumulator = Sum( Weight[feature] for feature in active_features ) + Bias

    Optimized using highly parallel embedding table lookups.
    """

    def __init__(
        self,
        feature_dim: int = CONFIG.FEATURE_DIM,
        accumulator_dim: int = CONFIG.ACCUMULATOR_DIM,
    ) -> None:
        super().__init__()

        self.feature_dim = feature_dim
        self.accumulator_dim = accumulator_dim

        # ------------------------------------------------------
        # Sparse Feature Embedding Matrix Lookup Table
        # Shape: [40960, 512]
        # ------------------------------------------------------
        self.embedding = nn.Embedding(
            num_embeddings=feature_dim,
            embedding_dim=accumulator_dim,
            sparse=False,
        )

        # ------------------------------------------------------
        # Transformed Accumulator View Accumulation Bias
        # Shape: [512]
        # ------------------------------------------------------
        self.bias = nn.Parameter(torch.zeros(accumulator_dim))
        self.reset_parameters()

    # ==========================================================
    # Layer Initialization
    # ==========================================================

    def reset_parameters(self) -> None:
        """Stable initialization distribution configuration targeting chess neural structures."""
        # Standard deviation normalized to prevent explosive gradient accumulation
        nn.init.normal_(
            self.embedding.weight,
            mean=0.0,
            std=0.01,
        )
        nn.init.zeros_(self.bias)

    # ==========================================================
    # Full Reconstruction (Single Position Inference/Fallback)
    # ==========================================================

    def forward(self, active_features: torch.Tensor) -> torch.Tensor:
        """
        Builds a dense accumulator completely from scratch for a single position.

        Parameters
        ----------
        active_features : Tensor[int64], Shape: [num_active_features]

        Returns
        -------
        accumulator : Tensor[float32], Shape: [accumulator_dim]
        """
        if active_features.numel() == 0:
            return self.bias.clone()

        # Lookup and sum active embeddings across the feature column axis
        embeddings = self.embedding(active_features)
        return embeddings.sum(dim=0) + self.bias

    # ==========================================================
    # Batched Parallel Reconstruction (Highly Optimized for Training)
    # ==========================================================

    def batch_forward(self, batch_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Builds accumulators for an entire batch concurrently.
        Uses advanced zero-padding vectorization to eliminate nested Python loops.

        Parameters
        ----------
        batch_features : List[Tensor[int64]], Length: [batch_size]

        Returns
        -------
        accumulators : Tensor[float32], Shape: [batch_size, accumulator_dim]
        """
        device = self.embedding.weight.device
        batch_size = len(batch_features)

        if batch_size == 0:
            return torch.empty((0, self.accumulator_dim), device=device)

        # Vectorized Padding: Track the maximum active feature length across the positions
        lengths = [f.numel() for f in batch_features]
        max_len = max(lengths)

        # Pre-allocate a 2D matrix filled with zero-indices: Shape [B, MaxFeatures]
        padded_tensor = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)

        # Populate indices and build an operations masking matrix
        mask = torch.zeros((batch_size, max_len, 1), dtype=torch.float32, device=device)
        for i, features in enumerate(batch_features):
            num_features = features.numel()
            if num_features > 0:
                padded_tensor[i, :num_features] = features
                mask[i, :num_features, :] = 1.0

        # Execute parallel matrix embedding lookup: Shape [B, MaxFeatures, AccumulatorDim]
        embedded = self.embedding(padded_tensor)

        # Apply mask to safely nullify padded values, then sum across the sequence dimension
        accumulators = (embedded * mask).sum(dim=1)

        # Inject the bias row broadcasted across the active batch dimension
        return accumulators + self.bias.unsqueeze(0)

    # ==========================================================
    # Incremental State Modifiers (Blazing Fast Live Engine Search)
    # ==========================================================

    @torch.no_grad()
    def add_features(
        self, accumulator: torch.Tensor, added_features: Iterable[int]
    ) -> torch.Tensor:
        """Incrementally add feature indices into the running accumulator."""
        features_list = list(added_features)
        if not features_list:
            return accumulator

        idx = torch.tensor(features_list, dtype=torch.long, device=accumulator.device)
        delta = self.embedding(idx).sum(dim=0)
        return accumulator + delta

    @torch.no_grad()
    def remove_features(
        self, accumulator: torch.Tensor, removed_features: Iterable[int]
    ) -> torch.Tensor:
        """Incrementally subtract feature indices from the running accumulator."""
        features_list = list(removed_features)
        if not features_list:
            return accumulator

        idx = torch.tensor(features_list, dtype=torch.long, device=accumulator.device)
        delta = self.embedding(idx).sum(dim=0)
        return accumulator - delta

    @torch.no_grad()
    def update_accumulator(
        self,
        accumulator: torch.Tensor,
        removed_features: Iterable[int],
        added_features: Iterable[int],
    ) -> torch.Tensor:
        """
        Executes a rapid, incremental update step on the active accumulator state.
        Ensures execution speed remains high during live search evaluations.
        """
        accumulator = self.remove_features(accumulator, removed_features)
        return self.add_features(accumulator, added_features)

    # ==========================================================
    # Quantization Precision Boundary Clamping
    # ==========================================================

    def clamp_weights(self) -> None:
        """Clamps parameter weights inside normal floating point bounds to prevent overflow."""
        with torch.no_grad():
            self.embedding.weight.clamp_(
                CONFIG.WEIGHT_CLAMP_MIN, CONFIG.WEIGHT_CLAMP_MAX
            )
            self.bias.clamp_(
                CONFIG.WEIGHT_CLAMP_MIN, CONFIG.WEIGHT_CLAMP_MAX
            )
