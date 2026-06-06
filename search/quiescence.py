# search/quiescence.py

from __future__ import annotations
import chess

# ==========================================================
# CONSTANTS
# ==========================================================

DELTA_MARGIN = 200

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}


# ==========================================================
# QUIESCENCE SEARCH
# ==========================================================

class QuiescenceSearch:
    """
    Negamax Quiescence Search safely wired into the NNUE stack framework.
    """

    def __init__(
        self,
        evaluator,         # Your PositionEvaluator
        move_ordering=None,
        enable_delta_pruning: bool = True,
    ) -> None:
        self.evaluator = evaluator
        self.move_ordering = move_ordering
        self.enable_delta_pruning = enable_delta_pruning
        self.nodes = 0

    def quiescence(self, board: chess.Board, alpha: int, beta: int) -> int:
        """
        Explores tactical branches until a quiet position is established.
        """
        self.nodes += 1

        # ----------------------------------
        # 1. Stand Pat Evaluation
        # ----------------------------------
        stand_pat = self.evaluator.evaluate(board)

        # Fail-high (Prune if current position is too good for opponent to allow)
        if stand_pat >= beta:
            return beta

        if stand_pat > alpha:
            alpha = stand_pat

        # ----------------------------------
        # 2. Generate and Order Noisy Moves
        # ----------------------------------
        moves = self.generate_tactical_moves(board)
        if not moves:
            return alpha

        if self.move_ordering is not None:
            moves = self.move_ordering.order_moves(board, moves, ply=99, tt_move=None)

        # ----------------------------------
        # 3. Search Loop
        # ----------------------------------
        for move in moves:

            # Delta Pruning Optimization
            if self.enable_delta_pruning and self._delta_prune(board, move, stand_pat, alpha):
                continue

            # --- SYNC NNUE FRAMEWORK STEP FORWARD ---
            self.evaluator.nnue.push(move)


            # Recursive tactical evaluation pass
            score = -self.quiescence(board, -beta, -alpha)

            # --- SYNC NNUE FRAMEWORK STEP BACK ---

            self.evaluator.nnue.pop()

            if score >= beta:
                return beta

            if score > alpha:
                alpha = score

        return alpha

    def generate_tactical_moves(self, board: chess.Board) -> list[chess.Move]:
        """ Optimized tactical collector targeting captures and promotions. """
        tactical_moves = []
        for move in board.legal_moves:
            # Captures or Promotions
            if board.is_capture(move) or move.promotion:
                tactical_moves.append(move)
                continue

            # Fallback check detection (optimized out of full push/pop trees)
            if board.gives_check(move):
                tactical_moves.append(move)

        return tactical_moves

    def _delta_prune(self, board: chess.Board, move: chess.Move, stand_pat: int, alpha: int) -> bool:
        """ Skips deep checking on obviously failed capture material gains. """
        if not board.is_capture(move):
            return False

        # En Passant handling fallback
        if board.is_en_passant(move):
            return False

        victim = board.piece_at(move.to_square)
        if victim is None:
            return False

        gain = PIECE_VALUES.get(victim.piece_type, 0)

        # If static position value + material capture value + buffer margin cannot break alpha, skip
        return (stand_pat + gain + DELTA_MARGIN) < alpha

    def reset(self) -> None:
        self.nodes = 0

    def __repr__(self) -> str:
        return f"QuiescenceSearch(nodes={self.nodes})"
