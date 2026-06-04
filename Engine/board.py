# engine/board.py
from __future__ import annotations

import chess
from typing import Optional, Union


class ChessBoard:
    """
    Thin wrapper around python-chess.Board.
    Responsibilities: State management, FEN IO, and Make/Undo.
    """

    def __init__(self, fen: Optional[str] = None) -> None:
        self._board = chess.Board(fen) if fen else chess.Board()

    def reset(self) -> None:
        self._board.reset()

    def load_fen(self, fen: str) -> None:
        self._board.set_fen(fen)

    def export_fen(self) -> str:
        return self._board.fen()

    def make_move(self, move: Union[str, chess.Move]) -> bool:
        if isinstance(move, str):
            try:
                move = chess.Move.from_uci(move)
            except ValueError:
                return False

        # Legality check is still performed here before pushing to the state
        if move not in self._board.legal_moves:
            return False

        self._board.push(move)
        return True

    def undo_move(self) -> Optional[chess.Move]:
        if not self._board.move_stack:
            return None
        return self._board.pop()

    def is_game_over(self) -> bool:
        return self._board.is_game_over()

    def get_result(self) -> str:
        return self._board.result(claim_draw=True)

    def get_turn(self) -> str:
        return "white" if self._board.turn == chess.WHITE else "black"

    def clone(self) -> "ChessBoard":
        cloned = ChessBoard()
        cloned._board = self._board.copy(stack=True)
        return cloned

    @property
    def halfmove_clock(self) -> int:
        """Number of halfmoves since the last pawn move or capture."""
        return self._board.halfmove_clock

    @property
    def fullmove_number(self) -> int:
        """The current fullmove number of the game."""
        return self._board.fullmove_number

    @property
    def en_passant_square(self) -> Optional[str]:
        """Returns en passant target square in algebraic notation (e.g. 'e3')."""
        if self._board.ep_square is None:
            return None
        return chess.square_name(self._board.ep_square)
    
    @property
    def board(self) -> chess.Board:
        """Exposes the underlying board for the MoveGenerator and Search layers."""
        return self._board

    def __str__(self) -> str:
        return str(self._board)

    def __repr__(self) -> str:
        return f"ChessBoard(fen='{self.export_fen()}')"
