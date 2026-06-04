# engine/move.py

from __future__ import annotations

import chess
from typing import Optional


class Move:
    """
    Lightweight wrapper around python-chess.Move.

    Responsibilities:
    - UCI serialization/deserialization
    - Move representation
    - Promotion support
    - Equality and hashing

    Notes:
    - No move legality checks are performed here.
    - Intended for heavy use in search trees.
    - Compatible with python-chess APIs.
    """

    __slots__ = ("_move",)

    def __init__(self, move: chess.Move) -> None:
        """
        Initialize from a python-chess Move object.

        Args:
            move: chess.Move instance.
        """
        self._move = move

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_uci(cls, uci: str) -> "Move":
        """
        Create Move from UCI string.

        Examples:
            e2e4
            g1f3
            e7e8q

        Args:
            uci: UCI move string.

        Returns:
            Move instance.

        Raises:
            ValueError:
                If UCI string is invalid.
        """
        return cls(chess.Move.from_uci(uci))

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_uci(self) -> str:
        """
        Convert move to UCI notation.

        Returns:
            UCI string.
        """
        return self._move.uci()

    # ------------------------------------------------------------------
    # Square Access
    # ------------------------------------------------------------------

    @property
    def from_square(self) -> int:
        """
        Source square index.

        Example:
            e2 -> 12
        """
        return self._move.from_square

    @property
    def to_square(self) -> int:
        """
        Destination square index.

        Example:
            e4 -> 28
        """
        return self._move.to_square

    # ------------------------------------------------------------------
    # Promotion
    # ------------------------------------------------------------------

    @property
    def promotion(self) -> Optional[int]:
        """
        Promotion piece type.

        Returns:
            chess.QUEEN
            chess.ROOK
            chess.BISHOP
            chess.KNIGHT
            or None
        """
        return self._move.promotion

    @property
    def is_promotion(self) -> bool:
        """
        Check whether move is a promotion move.
        """
        return self._move.promotion is not None

    # ------------------------------------------------------------------
    # Python-Chess Compatibility
    # ------------------------------------------------------------------

    @property
    def chess_move(self) -> chess.Move:
        """
        Access underlying python-chess Move.
        """
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