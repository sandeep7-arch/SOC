# nnue/model.py

from __future__ import annotations

import torch
import torch.nn as nn

from .config import CONFIG


class ClippedReLU(nn.Module):
    """
    NNUE-style clipped activation function layer.
    Computes clamp(x, min_value, max_value). Standard NNUE limits
    are bounded tightly between [0.0, 1.0] for efficient quantization.
    """

    def __init__(self, min_value: float = 0.0, max_value: float = 1.0) -> None:
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, self.min_value, self.max_value)


class NNUEModel(nn.Module):
    """
    Production-grade NNUE evaluation network.

    Input Topology:
        Concatenated Accumulators: [ACCUMULATOR_DIM * 2] (Typically 512 * 2 = 1024)
        Oriented dynamically relative to the side-to-move player.

    Internal Hidden Layout Topology:
        1024 (Dual L1) -> 256 (L2) -> 32 (L3) -> 1 (Out)
    """

    def __init__(self) -> None:
        super().__init__()

        self.accumulator_dim = CONFIG.ACCUMULATOR_DIM

        # Crucial Fix: Input dimension must accommodate both perspectives simultaneously
        self.hidden1 = nn.Linear(
            CONFIG.ACCUMULATOR_DIM * 2,
            CONFIG.HIDDEN1_DIM,
        )

        self.hidden2 = nn.Linear(
            CONFIG.HIDDEN1_DIM,
            CONFIG.HIDDEN2_DIM,
        )

        self.output = nn.Linear(
            CONFIG.HIDDEN2_DIM,
            CONFIG.OUTPUT_DIM,
        )

        self.activation = ClippedReLU(0.0, 1.0)
        self.reset_parameters()

    # ==========================================================
    # Initialization
    # ==========================================================

    def reset_parameters(self) -> None:
        """Stable parameter weight initialization for positional game play evaluation."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    # ==========================================================
    # Forward Pass
    # ==========================================================

    def forward(self, oriented_accumulator_pair: torch.Tensor) -> torch.Tensor:
        """
        Executes raw core matrix layer tensor operations.

        Parameters
        ----------
        oriented_accumulator_pair : Tensor[float32]
            Shape: [BatchSize, 1024] or [1024]
        """
        x = self.hidden1(oriented_accumulator_pair)
        x = self.activation(x)

        x = self.hidden2(x)
        x = self.activation(x)

        x = self.output(x)
        return x.squeeze(-1)

    # ==========================================================
    # Engine Search Evaluation Bridges (Single Position)
    # ==========================================================

    @torch.no_grad()
    def evaluate_perspective(
        self,
        white_acc: torch.Tensor,
        black_acc: torch.Tensor,
        is_white_to_move: bool,
    ) -> float:
        """Evaluates a single position from the active player's view axis."""
        # Enforce relative perspective: Side-to-move vector always comes first
        if is_white_to_move:
            oriented_input = torch.cat([white_acc, black_acc], dim=-1)
        else:
            oriented_input = torch.cat([black_acc, white_acc], dim=-1)

        score = self.forward(oriented_input)
        return float(score.item())

    @torch.no_grad()
    def evaluate_cp(
        self,
        white_acc: torch.Tensor,
        black_acc: torch.Tensor,
        is_white_to_move: bool,
    ) -> int:
        """
        Evaluates and converts raw values into integer centipawn bounds.
        This is the direct evaluation endpoint consumed by the search tree.
        """
        raw_score = self.evaluate_perspective(white_acc, black_acc, is_white_to_move)
        cp = int(raw_score * CONFIG.OUTPUT_SCALE)

        # Protect against extreme values using clamp boundaries
        return max(-CONFIG.MAX_CENTIPAWN_SCORE, min(CONFIG.MAX_CENTIPAWN_SCORE, cp))

    # ==========================================================
    # Batched Evaluation (Highly Optimized for Dataset Training)
    # ==========================================================

    def evaluate_batch(
        self,
        white_accumulators: torch.Tensor,
        black_accumulators: torch.Tensor,
        are_white_to_move: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluates a batch of positions concurrently during dataset training iterations.

        Parameters
        ----------
        white_accumulators : Tensor[float32], Shape: [BatchSize, 512]
        black_accumulators : Tensor[float32], Shape: [BatchSize, 512]
        are_white_to_move : Tensor[bool], Shape: [BatchSize]
        """
        batch_size = white_accumulators.shape[0]
        device = white_accumulators.device

        # Pre-allocate batched input tensor space
        oriented_inputs = torch.empty((batch_size, self.accumulator_dim * 2), dtype=torch.float32, device=device)

        # Build masking indices to apply parallel batch sorting
        w_mask = are_white_to_move == True
        b_mask = ~w_mask

        # Apply vectorized alignment based on perspective orientation masks
        if w_mask.any():
            oriented_inputs[w_mask] = torch.cat([white_accumulators[w_mask], black_accumulators[w_mask]], dim=-1)
        if b_mask.any():
            oriented_inputs[b_mask] = torch.cat([black_accumulators[b_mask], white_accumulators[b_mask]], dim=-1)

        return self.forward(oriented_inputs)

    # ==========================================================
    # Quantization Precision Boundary Clamping
    # ==========================================================

    def clamp_weights(self) -> None:
        """Clamps network parameters inside normal floating-point limits to prevent overflow."""
        with torch.no_grad():
            for param in self.parameters():
                param.clamp_(CONFIG.WEIGHT_CLAMP_MIN, CONFIG.WEIGHT_CLAMP_MAX)

    def quantization_ready(self) -> None:
        """Puts model into evaluation mode to lock tracking layers before exporting."""
        self.eval()

    # ==========================================================
    # Diagnostic Utilities
    # ==========================================================

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def architecture_summary(self) -> dict:
        return {
            "input_layer_dim": CONFIG.ACCUMULATOR_DIM * 2,
            "hidden_layer_1": CONFIG.HIDDEN1_DIM,
            "hidden_layer_2": CONFIG.HIDDEN2_DIM,
            "output_layer_dim": CONFIG.OUTPUT_DIM,
            "total_trainable_parameters": self.parameter_count(),
        }
