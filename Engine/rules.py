# engine/rules.py

from __future__ import annotations

import chess
from typing import Optional


def is_castling_move(move: chess.Move) -> bool:
    """
    Check whether a move is a castling move.

    Args:
        move:
            python-chess Move instance.

    Returns:
        True if move is kingside or queenside castling.
    """
    # Castling is represented as a king move
    # from e-file to g-file or c-file.
    return (
        chess.square_file(move.from_square) == 4
        and chess.square_rank(move.from_square)
        in (0, 7)
        and chess.square_file(move.to_square)
        in (2, 6)
    )


def is_en_passant_move(
    board: chess.Board,
    move: chess.Move,
) -> bool:
    """
    Check whether a move is an en passant capture.

    Args:
        board:
            Active python-chess Board.

        move:
            python-chess Move instance.

    Returns:
        True if move is en passant.
    """
    return board.is_en_passant(move)


def is_promotion_move(move: chess.Move) -> bool:
    """
    Check whether a move is a promotion.

    Args:
        move:
            python-chess Move instance.

    Returns:
        True if move promotes a pawn.
    """
    return move.promotion is not None


def get_promotion_piece(
    move: chess.Move,
) -> Optional[int]:
    """
    Return promotion piece type.

    Returns:
        chess.QUEEN
        chess.ROOK
        chess.BISHOP
        chess.KNIGHT
        or None
    """
    return move.promotion


def validate_special_move(
    board: chess.Board,
    move: chess.Move,
) -> bool:
    """
    Validate a special move using python-chess legality.

    Supported special moves:
    - Castling
    - En Passant
    - Promotion

    Notes:
    - No custom rule implementation.
    - Legality is delegated entirely to python-chess.

    Args:
        board:
            Active python-chess Board.

        move:
            python-chess Move instance.

    Returns:
        True if move is legal.
    """
    return move in board.legal_moves