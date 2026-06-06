# search/alpha_beta.py

from __future__ import annotations
import chess
from .transposition import TTFlag

# ==========================================================
# CONSTANTS
# ==========================================================

INF = 1_000_000
MATE_SCORE = 30_000


# ==========================================================
# SEARCH TIMEOUT
# ==========================================================

class SearchTimeout(Exception):
    """ Raised when the time manager requests an immediate search stop. """
    pass


# ==========================================================
# ALPHA BETA SEARCH
# ==========================================================

class AlphaBetaSearch:

    def __init__(
        self,
        evaluator,
        quiescence,
        move_ordering,
        transposition_table,
        time_manager=None,
    ) -> None:
        self.evaluator = evaluator
        self.quiescence = quiescence
        self.move_ordering = move_ordering
        self.tt = transposition_table
        self.time_manager = time_manager

        self.nodes = 0

    # ======================================================
    # PUBLIC ENTRY
    # ======================================================

    def search(
        self,
        board: chess.Board,
        depth: int,
        alpha: int = -INF,
        beta: int = INF,
        is_root: bool = False,
    ) -> tuple[int, chess.Move | None]:
        """
        Main search entry for an individual depth tier.
        """
        # Node tracking management is maintained across iterative deepening tiers.
        # It is cleared exclusively by the master search initialization pipeline.

        score, move = self._negamax(
            board=board,
            depth=depth,
            alpha=alpha,
            beta=beta,
            ply=0,
            is_root=is_root,
        )

        return score, move

    # ======================================================
    # NEGAMAX INTERNALS
    # ======================================================

    def _negamax(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        ply: int,
        is_root: bool = False,
    ) -> tuple[int, chess.Move | None]:
        """ Production-grade Negamax Alpha-Beta search. """
        self.nodes += 1

        # ----------------------------------
        # Throttled Time Check Management
        # ----------------------------------
        # Only query system clocks once every 2048 nodes to maximize search speed
        if (self.nodes & 2047) == 0:
            if self.time_manager is not None and self.time_manager.should_stop_search():
                raise SearchTimeout()

        original_alpha = alpha

        # ----------------------------------
        # Terminal Positions Evaluation
        # ----------------------------------
        if not is_root:
            terminal = self.evaluator.terminal_score(board, ply)
            if terminal is not None:
                return terminal, None

        # ----------------------------------
        # Quiescence Leaf Horizon Cutoff
        # ----------------------------------
        if depth <= 0:
            score = self.quiescence.quiescence(board, alpha, beta)
            return score, None

        # ----------------------------------
        # High-Speed Zobrist Native Lookup
        # ----------------------------------
        zobrist = self.evaluator.nnue._hash(board)

        # ----------------------------------
        # Transposition Table Probing
        # ----------------------------------
        tt_score = self.tt.probe(
            zobrist=zobrist,
            depth=depth,
            alpha=alpha,
            beta=beta,
            ply=ply,
        )
        if tt_score is not None:
            return tt_score, None

        tt_entry = self.tt.lookup(zobrist)
        tt_move = tt_entry.best_move if tt_entry is not None else None

        # ----------------------------------
        # Move Generation & Reordering Heuristics
        # ----------------------------------
        moves = self.move_ordering.order_moves(
            board,
            board.legal_moves,
            ply,
            tt_move,
        )

        if not moves:
            # Safe catch-all fallback for empty legal allocations
            return 0, None

        # ----------------------------------
        # Core Branch Search Execution Loop
        # ----------------------------------
        best_score = -INF
        best_move = None

        for move in moves:
            # Capture material state profile before pushing move state changes
            is_capture_move = board.is_capture(move)

            # --- SYNC NNUE FRAMEWORK STEP FORWARD ---
            self.evaluator.nnue.push(move)


            score, _ = self._negamax(
                board=board,
                depth=depth - 1,
                alpha=-beta,
                beta=-alpha,
                ply=ply + 1,
                is_root=False,
            )
            score = -score

            # --- SYNC NNUE FRAMEWORK STEP BACK ---

            self.evaluator.nnue.pop()

            # Track global alpha tracking best scoring metrics
            if score > best_score:
                best_score = score
                best_move = move

            if score > alpha:
                alpha = score

            # Beta Cutoff (Branch Pruning Trigger)
            if alpha >= beta:
                # Reward quiet moves that caused cutoffs
                if not is_capture_move:
                    self.move_ordering.add_killer(move, ply)
                    self.move_ordering.add_history(board.turn, move, depth)
                break

        # ----------------------------------
        # Store State in Transposition Table
        # ----------------------------------
        flag = TTFlag.EXACT
        if best_score <= original_alpha:
            flag = TTFlag.UPPERBOUND
        elif best_score >= beta:
            flag = TTFlag.LOWERBOUND

        self.tt.store(
            zobrist=zobrist,
            depth=depth,
            score=best_score,
            flag=flag,
            best_move=best_move,
            ply=ply,
        )

        return best_score, best_move

    # ======================================================
    # UTILITIES
    # ======================================================

    def reset(self) -> None:
        self.nodes = 0

    def get_node_count(self) -> int:
        return self.nodes

    def __repr__(self) -> str:
        return f"AlphaBetaSearch(nodes={self.nodes})"
