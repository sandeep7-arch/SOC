# engine/move.py
from __future__ import annotations

from typing import Optional
import chess


class Move:
    """
    Lightweight, immutable data container wrapping a python-chess.Move object.

    Responsibilities:
    - UCI syntax serialization and deserialization ('e2e4', 'g1f3').
    - Acts as a structured bridge token between internal layers and external boundaries.
    - Captures move properties (squares, promotions) for move-ordering routines.

    Strict Constraints:
    - Zero rule calculation, state alterations, or validation checks live here.
    - Memory-optimized via __slots__ to eliminate variable dictionary overhead.
    """

    __slots__ = ("_move",)

    def __init__(self, move: chess.Move) -> None:
        """Initialize and seal a python-chess Move object variant."""
        self._move = move

    # ------------------------------------------------------------------
    # Construction Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_uci(cls, uci: str) -> "Move":
        """
        Create a Move wrapper token out of a standard UCI coordinate string.
        Raises ValueError if the text syntax structure is malformed.
        """
        try:
            return cls(chess.Move.from_uci(uci))
        except ValueError as exc:
            raise ValueError(f"Malformed UCI move string format: {uci}") from exc

    @classmethod
    def from_squares(
        cls, from_square: int, to_square: int, promotion: Optional[int] = None
    ) -> "Move":
        """
        Construct a Move explicitly from board matrix coordinate index values (0-63).
        Highly effective for handling direct raw input or formatting custom test seeds.
        """
        return cls(chess.Move(from_square, to_square, promotion=promotion))

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_uci(self) -> str:
        """Convert the underlying move back into a canonical text-string identifier."""
        return self._move.uci()

    # ------------------------------------------------------------------
    # Coordinate Index Selectors
    # ------------------------------------------------------------------

    @property
    def from_square(self) -> int:
        """The absolute source square grid matrix index integer value (0 to 63)."""
        return self._move.from_square

    @property
    def to_square(self) -> int:
        """The absolute target square grid matrix index integer value (0 to 63)."""
        return self._move.to_square

    # ------------------------------------------------------------------
    # Functional Property Flags (Vital for Alpha-Beta Search Orderings)
    # ------------------------------------------------------------------

    @property
    def promotion(self) -> Optional[int]:
        """Returns the piece type identifier constant if a pawn promotes, else None."""
        return self._move.promotion

    @property
    def is_promotion(self) -> bool:
        """Boolean state flag showing if this action results in a piece promotion."""
        return self._move.promotion is not None

    @property
    def is_null(self) -> bool:
        """Returns True if this instance represents an empty null move pass sequence."""
        return not self._move

    # ------------------------------------------------------------------
    # Core Layer Unwrapping
    # ------------------------------------------------------------------

    @property
    def chess_move(self) -> chess.Move:
        """Direct access hook to retrieve the underlying raw python-chess structural object."""
        return self._move

    # ------------------------------------------------------------------
    # Identity Overrides
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Move):
            return False
        return self._move == other._move

    def __hash__(self) -> int:
        return hash(self._move)

    # ------------------------------------------------------------------
    # String Format Renderers
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        return self.to_uci()

    def __repr__(self) -> str:
        return f"Move('{self.to_uci()}')"
