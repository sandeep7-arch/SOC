# search/move_ordering.py

from __future__ import annotations
from typing import List, Iterable, Optional
import chess

# ==========================================================
# PIECE VALUES
# ==========================================================

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}


# ==========================================================
# MOVE ORDERING
# ==========================================================

class MoveOrdering:
    """
    Engine-grade move ordering implementation for Alpha-Beta search tree pruning.

    Priority Sorting Hierarchy:
        1. TT Move (Transposition recommendation)
        2. Promotion Captures (Tactical maximums)
        3. Quiet Promotions
        4. Captures (Most Valuable Victim - Least Valuable Attacker)
        5. Killer Moves (High-yielding quiet cutoffs)
        6. History Heuristic (Contextual positional value)
        7. Remaining Quiet Moves
    """

    MAX_PLY = 128

    TT_MOVE_BONUS = 10_000_000
    PROMOTION_BONUS = 9_000_000
    WINNING_CAPTURE_BONUS = 8_000_000
    KILLER_BONUS = 7_000_000

    def __init__(self) -> None:
        # Pre-allocate array slots to maximize searching throughput speed
        self.killer_moves: List[List[Optional[chess.Move]]] = [
            [None, None] for _ in range(self.MAX_PLY)
        ]

        # history[color][from_square][to_square]
        self.history = {
            chess.WHITE: [[0] * 64 for _ in range(64)],
            chess.BLACK: [[0] * 64 for _ in range(64)],
        }

    # ======================================================
    # PUBLIC SORTING API
    # ======================================================

    def order_moves(
        self,
        board: chess.Board,
        moves: Iterable[chess.Move],
        ply: int = 0,
        tt_move: Optional[chess.Move] = None,
    ) -> List[chess.Move]:
        """
        Sorts an iterable list of moves using heuristic priority scoring.
        """
        scored_moves = []

        for move in moves:
            score = self.score_move(
                board=board,
                move=move,
                ply=ply,
                tt_move=tt_move,
            )
            scored_moves.append((score, move))

        # Sort in-place using descending order (highest scoring moves checked first)
        scored_moves.sort(key=lambda item: item[0], reverse=True)

        return [move for _, move in scored_moves]

    # ======================================================
    # CORE MOVE SCORING SYSTEM
    # ======================================================

    def score_move(
        self,
        board: chess.Board,
        move: chess.Move,
        ply: int = 0,
        tt_move: Optional[chess.Move] = None,
    ) -> int:
        """
        Assigns static prioritization weights to a specific candidate move.
        """
        # ----------------------------------
        # 1. Transposition Table Cutoff Move
        # ----------------------------------
        if tt_move is not None and move == tt_move:
            return self.TT_MOVE_BONUS

        is_capture = board.is_capture(move)
        score = 0

        # ----------------------------------
        # 2. Promotion Handling
        # ----------------------------------
        if move.promotion:
            promoted_value = PIECE_VALUES.get(move.promotion, 0)
            score += self.PROMOTION_BONUS + promoted_value
            # If it's a promotion-capture, add tactical weight to sort it first
            if is_capture:
                score += self._mvv_lva_score(board, move) + 500_000
            return score

        # ----------------------------------
        # 3. Standard Captures (MVV-LVA)
        # ----------------------------------
        if is_capture:
            score += self.WINNING_CAPTURE_BONUS
            score += self._mvv_lva_score(board, move)
            return score

        # ----------------------------------
        # 4. Killer Moves
        # ----------------------------------
        if ply < self.MAX_PLY:
            killers = self.killer_moves[ply]
            if move == killers[0]:
                return self.KILLER_BONUS + 1000
            if move == killers[1]:
                return self.KILLER_BONUS

        # ----------------------------------
        # 5. History Heuristic & Quiet Moves
        # ----------------------------------
        color = board.turn
        score += self.history[color][move.from_square][move.to_square]
        return score

    # ======================================================
    # MVV-LVA ENGINE
    # ======================================================

    def _mvv_lva_score(self, board: chess.Board, move: chess.Move) -> int:
        """
        Calculates Most Valuable Victim / Least Valuable Attacker prioritization values.
        """
        attacker_piece = board.piece_at(move.from_square)
        if attacker_piece is None:
            return 0

        # Fix En Passant target square lookup discrepancy natively
        if board.is_en_passant(move):
            victim_value = PIECE_VALUES[chess.PAWN]
        else:
            victim_piece = board.piece_at(move.to_square)
            victim_value = PIECE_VALUES[victim_piece.piece_type] if victim_piece else 0

        attacker_value = PIECE_VALUES[attacker_piece.piece_type]

        # Scale victim value higher so low-value pieces capturing high-value targets rank first
        return (victim_value * 10) - attacker_value

    # ======================================================
    # REWARD UPDATES
    # ======================================================

    def add_killer(self, move: chess.Move, ply: int) -> None:
        """ Stores a high-performing quiet move that caused a beta cutoff. """
        if ply >= self.MAX_PLY:
            return

        killers = self.killer_moves[ply]
        if move == killers[0]:
            return

        # Shift primary slot out to auxiliary slot
        killers[1] = killers[0]
        killers[0] = move

    def add_history(self, color: bool, move: chess.Move, depth: int) -> None:
        """ Appends weights to quiet positioning paths based on successful cutoffs. """
        bonus = depth * depth
        self.history[color][move.from_square][move.to_square] += bonus

    def decay_history(self) -> None:
        """ Stabilizes variable history ranges using an integer bitwise-style shift. """
        for color in (chess.WHITE, chess.BLACK):
            table = self.history[color]
            for frm in range(64):
                row = table[frm]
                for to in range(64):
                    row[to] //= 2

    def clear(self) -> None:
        """ Resets temporary evaluation cache states. """
        self.killer_moves = [[None, None] for _ in range(self.MAX_PLY)]
        self.history = {
            chess.WHITE: [[0] * 64 for _ in range(64)],
            chess.BLACK: [[0] * 64 for _ in range(64)],
        }

    def killer_count(self) -> int:
        return sum(1 for pair in self.killer_moves for move in pair if move is not None)

    def __repr__(self) -> str:
        return f"MoveOrdering(killers={self.killer_count()})"
