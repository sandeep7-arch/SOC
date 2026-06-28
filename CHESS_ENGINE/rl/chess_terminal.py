"""Terminal chess rules used by Python RL orchestration."""

from __future__ import annotations

from collections.abc import Mapping

from uci import UciPosition


MATE_SCORE_CP = 4000


def repetition_key(position: UciPosition) -> str:
    """Return the FEN fields relevant for repetition detection."""
    return " ".join(position.to_fen().split()[:4])


def terminal_draw_reason(
    position: UciPosition,
    repetitions: Mapping[str, int] | None = None,
) -> str | None:
    if position.halfmove_clock >= 100:
        return "Draw: fifty-move rule"
    if repetitions is not None and repetitions.get(repetition_key(position), 0) >= 3:
        return "Draw: threefold repetition"
    if has_insufficient_material(position):
        return "Draw: insufficient material"
    return None


def terminal_result_for_no_move(position: UciPosition) -> tuple[float, str]:
    side_name = "white" if position.turn == "w" else "black"
    if side_to_move_is_in_check(position):
        result = -float(MATE_SCORE_CP) if position.turn == "w" else float(MATE_SCORE_CP)
        return result, f"Checkmate: no legal move for {side_name}"
    return 0.0, f"Stalemate: no legal move for {side_name}"


def model_score_for_no_move(position: UciPosition, model_to_move: bool) -> float:
    if side_to_move_is_in_check(position):
        return -1.0 if model_to_move else 1.0
    return 0.0


def has_insufficient_material(position: UciPosition) -> bool:
    pieces = [piece for piece in position.board if piece is not None]
    non_kings = [piece for piece in pieces if piece.lower() != "k"]
    if not non_kings:
        return True
    if any(piece.lower() in ("p", "r", "q") for piece in non_kings):
        return False
    if len(non_kings) == 1:
        return non_kings[0].lower() in ("b", "n")
    if all(piece.lower() == "b" for piece in non_kings):
        bishop_colors = {
            (index % 8 + index // 8) % 2
            for index, piece in enumerate(position.board)
            if piece is not None and piece.lower() == "b"
        }
        return len(bishop_colors) == 1
    return False


def side_to_move_is_in_check(position: UciPosition) -> bool:
    king = "K" if position.turn == "w" else "k"
    try:
        king_square = position.board.index(king)
    except ValueError:
        return False
    attacker = "b" if position.turn == "w" else "w"
    return is_square_attacked(position, king_square, attacker)


def is_square_attacked(position: UciPosition, square: int, attacker: str) -> bool:
    for source, piece in enumerate(position.board):
        if piece is None:
            continue
        if (piece.isupper() and attacker != "w") or (piece.islower() and attacker != "b"):
            continue
        if _piece_attacks_square(position, source, piece, square):
            return True
    return False


def _piece_attacks_square(position: UciPosition, source: int, piece: str, target: int) -> bool:
    source_file = source % 8
    source_rank = source // 8
    target_file = target % 8
    target_rank = target // 8
    file_delta = target_file - source_file
    rank_delta = target_rank - source_rank
    lower = piece.lower()

    if lower == "p":
        direction = 1 if piece.isupper() else -1
        return rank_delta == direction and abs(file_delta) == 1
    if lower == "n":
        return (abs(file_delta), abs(rank_delta)) in ((1, 2), (2, 1))
    if lower == "k":
        return max(abs(file_delta), abs(rank_delta)) == 1
    if lower == "b":
        return abs(file_delta) == abs(rank_delta) and _clear_ray(position, source, target)
    if lower == "r":
        return (file_delta == 0 or rank_delta == 0) and _clear_ray(position, source, target)
    if lower == "q":
        straight = file_delta == 0 or rank_delta == 0
        diagonal = abs(file_delta) == abs(rank_delta)
        return (straight or diagonal) and _clear_ray(position, source, target)
    return False


def _clear_ray(position: UciPosition, source: int, target: int) -> bool:
    source_file = source % 8
    source_rank = source // 8
    target_file = target % 8
    target_rank = target // 8
    file_step = _sign(target_file - source_file)
    rank_step = _sign(target_rank - source_rank)
    file_index = source_file + file_step
    rank_index = source_rank + rank_step
    while (file_index, rank_index) != (target_file, target_rank):
        if position.board[rank_index * 8 + file_index] is not None:
            return False
        file_index += file_step
        rank_index += rank_step
    return True


def _sign(value: int) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0

