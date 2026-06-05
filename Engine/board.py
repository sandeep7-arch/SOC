# engine/board.py
from __future__ import annotations

import chess
from typing import Optional, Union


class ChessBoard:
    """
    Authoritative state storage machine wrapping python-chess.Board.

    Responsibilities:
    - Pure position updates (push / pop)
    - FEN data serialization and parsing
    - Board mutations (reset / clone)
    - Low-level data attribute accessors

    Strict Constraints:
    - Zero rule analysis logic (No game over, checkmate, or draw validation).
    - Acts entirely as a data vault for other operational modules.
    """

    def __init__(self, fen: Optional[str] = None) -> None:
        if fen and not self.is_valid_fen(fen):
            raise ValueError(f"Invalid initial FEN: {fen}")
        self._board = chess.Board(fen) if fen else chess.Board()

    def reset(self) -> None:
        """Reset the internal board back to the standard starting layout."""
        self._board.reset()

    # ------------------------------------------------------------------
    # FEN Serializations
    # ------------------------------------------------------------------

    def load_fen(self, fen: str) -> bool:
        """Loads a FEN string. Returns False if structurally invalid."""
        if not self.is_valid_fen(fen):
            return False
        self._board.set_fen(fen)
        return True

    def export_fen(self) -> str:
        """Export the exact active layout snapshot as a canonical FEN string."""
        return self._board.fen()

    @staticmethod
    def is_valid_fen(fen_str: str) -> bool:
        """Fast bitboard validation of a FEN syntax string without memory allocation."""
        return chess.Board.is_valid(fen_str)

    @staticmethod
    def get_starting_fen() -> str:
        """Returns the default chess starting layout FEN string."""
        return chess.STARTING_FEN

    # ------------------------------------------------------------------
    # State Mutations (Verbs)
    # ------------------------------------------------------------------

    def make_move(self, move: Union[str, chess.Move]) -> bool:
        """
        Executes a move onto the board state stack.
        Performs a rapid validation filter check before mutating state.
        """
        if isinstance(move, str):
            try:
                move = chess.Move.from_uci(move)
            except ValueError:
                return False

        if move not in self._board.legal_moves:
            return False

        self._board.push(move)
        return True

    def undo_move(self) -> Optional[chess.Move]:
        """Reverts the last move played on the stack. Returns None if history empty."""
        if not self._board.move_stack:
            return None
        return self._board.pop()

    def clone(self) -> "ChessBoard":
        """Generates a deep, independent memory clone of the entire match state."""
        cloned = ChessBoard()
        cloned._board = self._board.copy(stack=True)
        return cloned

    # ------------------------------------------------------------------
    # State Primitives (Data Accessors)
    # ------------------------------------------------------------------

    def get_turn_color(self) -> str:
        """Returns string representation of the active player color."""
        return "white" if self._board.turn == chess.WHITE else "black"

    @property
    def halfmove_clock(self) -> int:
        """Tracks clock cycles since last capture/pawn push for fifty-move draw checks."""
        return self._board.halfmove_clock

    @property
    def fullmove_number(self) -> int:
        """The absolute fullmove counter index of the active game."""
        return self._board.fullmove_number

    @property
    def en_passant_square(self) -> Optional[str]:
        """Returns the string square label of a valid en passant ghost capture square."""
        if self._board.ep_square is None:
            return None
        return chess.square_name(self._board.ep_square)

    # ------------------------------------------------------------------
    # Castling Rights Storage API
    # ------------------------------------------------------------------

    @property
    def has_white_kingside_rights(self) -> bool:
        return self._board.has_kingside_castling_rights(chess.WHITE)

    @property
    def has_white_queenside_rights(self) -> bool:
        return self._board.has_queenside_castling_rights(chess.WHITE)

    @property
    def has_black_kingside_rights(self) -> bool:
        return self._board.has_kingside_castling_rights(chess.BLACK)

    @property
    def has_black_queenside_rights(self) -> bool:
        return self._board.has_queenside_castling_rights(chess.BLACK)

    @property
    def castling_rights_tuple(self) -> tuple[bool, bool, bool, bool]:
        """Returns all permission flags as a lightweight memory-fixed tuple."""
        return (
            self._board.has_kingside_castling_rights(chess.WHITE),
            self._board.has_queenside_castling_rights(chess.WHITE),
            self._board.has_kingside_castling_rights(chess.BLACK),
            self._board.has_queenside_castling_rights(chess.BLACK),
        )

    @property
    def board(self) -> chess.Board:
        """Direct access hook to underlying raw bitboard state for AI core engines."""
        return self._board

    def __str__(self) -> str:
        return str(self._board)

    def __repr__(self) -> str:
        return f"ChessBoard(fen='{self.export_fen()}')"
