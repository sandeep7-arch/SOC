# nnue/inference.py

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import chess
import torch

from .accumulator import Accumulator
from .feature_encoder import ENCODER
from .feature_transformer import FeatureTransformer
from .model import NNUEModel
from .zobrist import ZobristHasher


class NNUEEvaluator:
    """
    High-performance inference engine interface.
    Acts as the master facade. Your search layers interact ONLY with this class.

    Responsibilities:
        - Checkpoint model weight serialization loading.
        - Board evaluations (both incremental search state and stateless variants).
        - Stack allocation layer management.
        - Evaluation Transposition Cache tracking.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        use_cache: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.encoder = ENCODER

        # Instantiating key neural architecture layers
        self.transformer = FeatureTransformer().to(self.device)
        self.model = NNUEModel().to(self.device)
        self.model.eval()

        # Wire up the incremental tracking stack
        self.accumulator = Accumulator(
            encoder=self.encoder,
            transformer=self.transformer,
            device=device,
        )

        self.use_cache = use_cache
        self.cache: dict[int, int] = {}

        if model_path is not None:
            self.load_model(model_path)

    # ==========================================================
    # Model Loading
    # ==========================================================

    def load_model(self, path: str | Path) -> None:
        """Loads and splits weights into their respective architectural layers."""
        checkpoint = torch.load(path, map_location=self.device)

        if isinstance(checkpoint, dict) and ("model_state_dict" in checkpoint or "transformer_state_dict" in checkpoint):
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            if "transformer_state_dict" in checkpoint:
                self.transformer.load_state_dict(checkpoint["transformer_state_dict"])
        else:
            # Fallback for plain unified dictionary configurations
            self.model.load_state_dict(checkpoint)

        self.model.eval()

    # ==========================================================
    # Initialization
    # ==========================================================

    def initialize(self, board: chess.Board) -> None:
        """Sets up the initial state of the accumulator stack for a new position."""
        self.accumulator.initialize(board)

    # ==========================================================
    # Stateless Evaluation (Slower, For Out-of-Search Queries)
    # ==========================================================

    @torch.no_grad()
    def evaluate(self, board: chess.Board) -> int:
        """
        Runs an absolute standalone evaluation on an arbitrary board state.
        Bypasses running state stacks by assembling tensors from scratch.
        """
        zobrist = self._hash(board)
        if self.use_cache and zobrist in self.cache:
            return self.cache[zobrist]

        # Fix: Unpack the dual-perspective lists from our encoder
        w_features, b_features = self.encoder.encode(board)

        w_tensor = torch.tensor(w_features, dtype=torch.long, device=self.device)
        b_tensor = torch.tensor(b_features, dtype=torch.long, device=self.device)

        # Build individual accumulator representations
        w_acc = self.transformer(w_tensor)
        b_acc = self.transformer(b_tensor)

        # Evaluate position relative to the active turn perspective orientation
        is_white_to_move = (board.turn == chess.WHITE)
        score = self.model.evaluate_cp(w_acc, b_acc, is_white_to_move)

        if self.use_cache:
            self.cache[zobrist] = score

        return score

    # ==========================================================
    # Incremental Search Evaluation (Blazing Fast Alpha-Beta Hook)
    # ==========================================================

    @torch.no_grad()
    def evaluate_current(self) -> int:
    """Evaluates the current state vectors stored on the stack with cache lookup."""
        if self.accumulator.board is None:
            raise RuntimeError("Cannot evaluate uninitialized search position state.")

    # High-Performance Transposition Cache Lookup
        if self.use_cache:
            zobrist = self._hash(self.accumulator.board)
            if zobrist in self.cache:
                return self.cache[zobrist]

    # Compute if it's a cache miss
        w_acc, b_acc = self.accumulator.current_accumulators()
        is_white_to_move = (self.accumulator.board.turn == chess.WHITE)

        score = self.model.evaluate_cp(w_acc, b_acc, is_white_to_move)

    # Save to cache
        if self.use_cache:
            self.cache[zobrist] = score

        return score

    # ==========================================================
    # Search Push / Pop Unrolling
    # ==========================================================

    def push(self, move: chess.Move) -> int:
        """Steps state parameters forward incrementally and returns the score."""
        self.accumulator.push(move)
        return self.evaluate_current()

    def pop(self) -> None:
        """Reverts internal tracking parameters back by one historical state node."""
        self.accumulator.pop()

    # ==========================================================
    # Batched Evaluation (Dataset Processing Loop Optimization)
    # ==========================================================

    @torch.no_grad()
    def batch_evaluate(self, boards: Iterable[chess.Board]) -> List[int]:
        """Evaluates a collection of random positions using highly parallel execution batching."""
        w_accumulators_list = []
        b_accumulators_list = []
        turns_list = []

        for board in boards:
            w_features, b_features = self.encoder.encode(board)

            w_tensor = torch.tensor(w_features, dtype=torch.long, device=self.device)
            b_tensor = torch.tensor(b_features, dtype=torch.long, device=self.device)

            w_accumulators_list.append(self.transformer(w_tensor))
            b_accumulators_list.append(self.transformer(b_tensor))
            turns_list.append(torch.tensor(board.turn == chess.WHITE, device=self.device))

        if not w_accumulators_list:
            return []

        # Vectorize lists into clean multi-dimensional uniform matrices
        batch_w = torch.stack(w_accumulators_list)
        batch_b = torch.stack(b_accumulators_list)
        batch_turns = torch.stack(turns_list)

        # Process the matrices through our optimized model batch evaluation channel
        raw_outputs = self.model.evaluate_batch(batch_w, batch_b, batch_turns)

        # Scale outputs from raw floating point limits to centipawn integers
        from .config import CONFIG
        scaled_outputs = raw_outputs * CONFIG.OUTPUT_SCALE
        clamped_outputs = torch.clamp(scaled_outputs, -CONFIG.MAX_CENTIPAWN_SCORE, CONFIG.MAX_CENTIPAWN_SCORE)

        return [int(score.item()) for score in clamped_outputs]

    # ==========================================================
    # Transposition Cache Housekeeping
    # ==========================================================

    def clear_cache(self) -> None:
        self.cache.clear()

    def cache_size(self) -> int:
        return len(self.cache)

    def _hash(self, board: chess.Board) -> int:
    """
    Fetches the ultra-fast, incrementally updated hash tracking key.
    Bypasses expensive string generation overhead.
    """

        return ZobristHasher.get_native_hash(board)

    # ==========================================================
    # Hardware Device Migration
    # ==========================================================

    def to(self, device: str) -> None:
        """Migrates processing parameters to target hardware architectures."""
        self.device = torch.device(device)
        self.model.to(self.device)
        self.transformer.to(self.device)
        self.accumulator.device = self.device
