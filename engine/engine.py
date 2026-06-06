from __future__ import annotations
import chess
from typing import Tuple, Optional, Any

# Import Neural Infrastructure
from nnue.inference import NNUEEvaluator

# Import Search Infrastructure
from search.evaluator import PositionEvaluator
from search.transposition import TranspositionTable
from search.move_ordering import MoveOrdering
from search.quiescence import QuiescenceSearch
from search.alpha_beta import AlphaBetaSearch
from search.time_manager import TimeManager
from search.iterative_deepening import IterativeDeepeningSearch


class ChessEngine:
    """
    The master orchestrator class uniting the NNUE inference pipeline
    with the Alpha-Beta iterative deepening search framework.
    """

    def __init__(
        self,
        model_path: str,
        tt_entries: int = 2_000_000,
        eval_cache_size: int = 500_000
    ) -> None:
        print(f"info string Initializing Engine Components...")

        # 1. Initialize Neural Network Layer
        self.nnue = NNUEEvaluator(model_path=model_path)

        # 2. Initialize Core Subsystems
        self.tt = TranspositionTable(max_entries=tt_entries)
        self.move_ordering = MoveOrdering()
        self.time_manager = TimeManager()

        # 3. Build the Evaluation Bridge (Pass NNUE into it)
        self.evaluator = PositionEvaluator(
            nnue_evaluator=self.nnue,
            cache_size=eval_cache_size
        )

        # 4. Assemble the Search Stack Components
        # Note: Move ordering is shared by both full search and quiescence
        self.quiescence = QuiescenceSearch(
            evaluator=self.evaluator,
            move_ordering=self.move_ordering,
            enable_delta_pruning=True
        )

        self.alpha_beta = AlphaBetaSearch(
            evaluator=self.evaluator,
            quiescence=self.quiescence,
            move_ordering=self.move_ordering,
            transposition_table=self.tt,
            time_manager=self.time_manager
        )

        # 5. Top-Level Driver Framework
        self.search_driver = IterativeDeepeningSearch(
            searcher=self.alpha_beta,
            time_manager=self.time_manager,
            use_aspiration_windows=True
        )

        print(f"info string Engine successfully loaded model: {model_path}")

    # ======================================================
    # PRIMARY EXECUTION INTERFACE
    # ======================================================

    def play(
        self,
        board: chess.Board,
        time_limit: Optional[float] = None,
        max_depth: int = 64
    ) -> Tuple[Optional[chess.Move], int]:
        """
        Calculates the absolute best move for the current position.

        Returns:
            Tuple[chess.Move | None, int]: (Best candidate move, Evaluation score)
        """
        # Ensure search state and node counters are cleanly reset for a fresh move
        self.search_driver.reset()
        self.alpha_beta.reset()
        self.quiescence.reset()

        # Execute Iterative Deepening loop
        best_move, score, depth = self.search_driver.find_best_move(
            board=board,
            time_limit=time_limit,
            max_depth=max_depth
        )

        return best_move, score

    # ======================================================
    # LIFECYCLE & UCI SESSION MANAGERS
    # ======================================================

    def reset_game(self) -> None:
        """
        Flushes all lookahead histories and transposition entries clean.
        Must be called when receiving a UCI 'ucinewgame' command to prevent
        cross-game cache contamination.
        """
        print("info string Clearing transposition tables and engine memory slate...")
        self.tt.clear()                  # Fast C-Speed list multiplication wipe
        self.evaluator.clear_cache()     # Wipe standalone evaluation cache entries
        self.move_ordering.clear()       # Reset history heuristic weights & killer tables
        self.time_manager.reset()

    def get_diagnostics(self) -> dict[str, Any]:
        """
        Compiles structural performance data across the active subsystems.
        """
        return {
            "total_nodes": self.search_driver.total_nodes,
            "transposition_stats": self.tt.stats(),
            "evaluation_cache": self.evaluator.cache_stats(),
            "killer_slots_allocated": self.move_ordering.killer_count()
        }

    def __repr__(self) -> str:
        return (f"ChessEngine(TT_Capacity={self.tt.capacity()}, "
                f"EvalCache={self.evaluator.cache_size})")
