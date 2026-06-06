# search/evaluator.py

from __future__ import annotations
from typing import Optional
import chess

# ==========================================================
# SEARCH CONSTANTS
# ==========================================================

MATE_SCORE = 30000
MATE_THRESHOLD = 29000


# ==========================================================
# POSITION EVALUATOR
# ==========================================================

class PositionEvaluator:
    """
    Bridge layer connecting the Alpha-Beta Search tree to the NNUE Inference engine.

    Coordinates raw centipawn conversions, terminal game states, and
    maintains an optimized static evaluation cache to maximize search pruning throughput.
    """

    def __init__(
        self,
        nnue_evaluator,  # Instance of NNUEEvaluator from nnue/inference.py
        cache_size: int = 500_000,
    ) -> None:
        self.nnue = nnue_evaluator

        # Centralized Search Evaluation Cache
        self.cache_size = cache_size
        self.cache: dict[int, int] = {}

        self.cache_hits = 0
        self.cache_misses = 0

    # ======================================================
    # MAIN EVALUATION API
    # ======================================================

    def evaluate(self, board: chess.Board) -> int:
        """
        Evaluate a chess board position.

        Returns:
            int: Centipawn score from the perspective of the side-to-move.
                 Positive = Active side advantage
                 Negative = Opponent side advantage
        """
        # Fetch the fast Zobrist key from python-chess natively via your NNUE helper
        key = self.nnue._hash(board)

        # ----------------------------------
        # Cache Lookup
        # ----------------------------------
        cached = self.cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached

        self.cache_misses += 1

        # ----------------------------------
        # NNUE Inference
        # ----------------------------------
        # Call evaluate_current() if board is part of the search stack,
        # otherwise fall back to standalone evaluation.
        if self.nnue.accumulator.board == board:
            score = self.nnue.evaluate_current()
        else:
            score = self.nnue.evaluate(board)

        # ----------------------------------
        # Safety Clamp & Store
        # ----------------------------------
        score = self._normalize_score(score)

        if len(self.cache) >= self.cache_size:
            self._evict_cache_bulk()

        self.cache[key] = score
        return score

    # ======================================================
    # SCORE NORMALIZATION
    # ======================================================

    def _normalize_score(self, score: int) -> int:
        """
        Ensures the raw NNUE output does not bleed into the designated mate range.
        Mate tracking must remain uniquely owned by the search tree.
        """
        if score >= MATE_THRESHOLD:
            return MATE_THRESHOLD - 1
        elif score <= -MATE_THRESHOLD:
            return -MATE_THRESHOLD + 1
        return score

    # ======================================================
    # MATE SCORE HELPERS
    # ======================================================

    @staticmethod
    def mate_score(ply: int) -> int:
        """Returns winning mate score adjusted by current depth ply."""
        return MATE_SCORE - ply

    @staticmethod
    def mated_score(ply: int) -> int:
        """Returns losing mate score adjusted by current depth ply."""
        return -MATE_SCORE + ply

    # ======================================================
    # BULK CACHE EVICTION
    # ======================================================

    def _evict_cache_bulk(self) -> None:
        """
        Performs high-performance bulk clearing of old cache entries.
        Clears out a block of entries at once to prevent micro-eviction thrashing.
        """
        # Wipe 25% of the cache clean at once when limits are breached
        entries_to_remove = self.cache_size // 4

        # Python dicts maintain insertion order, so this naturally drops oldest inputs first
        keys_to_drop = list(self.cache.keys())[:entries_to_remove]
        for key in keys_to_drop:
            self.cache.pop(key, None)

    def clear_cache(self) -> None:
        """Completely flushes out the evaluation lookup tables."""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

    def cache_stats(self) -> dict:
        """Gathers runtime pipeline diagnostic performance analytics."""
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total) if total > 0 else 0.0

        return {
            "entries": len(self.cache),
            "cache_size": self.cache_size,
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": hit_rate,
        }

    # ======================================================
    # TERMINAL EVALUATION
    # ======================================================

    def terminal_score(self, board: chess.Board, ply: int) -> Optional[int]:
        """
        Evaluates strict endgame rule variations.

        Returns:
            An integer score if the game is over at this node, otherwise None.
        """
        if board.is_checkmate():
            return self.mated_score(ply)

        # Optimization: Check repetition at 2-plies deep to prevent infinite loops early
        if (
            board.is_stalemate()
            or board.is_insufficient_material()
            or board.is_fifty_moves()
            or board.is_repetition(count=2)
        ):
            return 0

        return None

    # ======================================================
    # DEBUG
    # ======================================================

    def __len__(self) -> int:
        return len(self.cache)

    def __repr__(self) -> str:
        return f"PositionEvaluator(cache_entries={len(self.cache)})"
