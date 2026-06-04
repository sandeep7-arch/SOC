from __future__ import annotations

import chess
from typing import Optional


class Move:
    """
    Lightweight wrapper around python-chess.Move.

    Responsibilities:
    - UCI serialization/deserialization
    - Move representation and promotion support
    - Equality, hashing, and border API translation

    Notes:
    - No move legality checks are performed here.
    - Optimized with __slots__ for boundary translation efficiency.
    """

    __slots__ = ("_move",)

    def __init__(self, move: chess.Move) -> None:
        """
        Initialize from a python-chess Move object.
        """
        self._move = move

    # ------------------------------------------------------------------
    # Construction Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_uci(cls, uci: str) -> "Move":
        """
        Create Move from UCI string (e.g., 'e2e4', 'e7e8q').
        """
        return cls(chess.Move.from_uci(uci))

    @classmethod
    def from_squares(cls, from_square: int, to_square: int, promotion: Optional[int] = None) -> "Move":
        """
        Create a Move explicitly from square indices.
        Useful for building user input moves or engine test suites.
        """
        return cls(chess.Move(from_square, to_square, promotion=promotion))

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_uci(self) -> str:
        """Convert move to UCI notation string."""
        return self._move.uci()

    # ------------------------------------------------------------------
    # Square Access
    # ------------------------------------------------------------------

    @property
    def from_square(self) -> int:
        """Source square index (0 to 63)."""
        return self._move.from_square

    @property
    def to_square(self) -> int:
        """Destination square index (0 to 63)."""
        return self._move.to_square

    # ------------------------------------------------------------------
    # Move Properties (Invaluable for move translation/ordering)
    # ------------------------------------------------------------------

    @property
    def promotion(self) -> Optional[int]:
        """Promotion piece type constant, or None."""
        return self._move.promotion

    @property
    def is_promotion(self) -> bool:
        """Check whether move is a promotion move."""
        return self._move.promotion is not None

    @property
    def null_move(self) -> bool:
        """Checks if this is a null (empty) move pass."""
        return not self._move

    # ------------------------------------------------------------------
    # Python-Chess Compatibility
    # ------------------------------------------------------------------

    @property
    def chess_move(self) -> chess.Move:
        """Access underlying python-chess Move object directly."""
        return self._move

    # ------------------------------------------------------------------
    # Equality / Hashing
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Move):
            return False
        return self._move == other._move

    def __hash__(self) -> int:
        return hash(self._move)

    # ------------------------------------------------------------------
    # String Representation
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        return self.to_uci()

    def __repr__(self) -> str:
        return f"Move('{self.to_uci()}')"
