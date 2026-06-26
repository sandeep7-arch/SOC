import chess
from blunder_detector import blunder_detector
from tactical_patterns import analyze_tactical_patterns
from king_safety import analyze_king_safety
from game_phase import get_game_phase

def categorize_mistake(board_before, board_after, eval_before, eval_after, turn):
    blunder_info = blunder_detector(eval_before, eval_after, turn)

    if blunder_info["classification"] == "good":
        return {
            "classification": "good",
            "category": None,
            "reason": None
        }

    phase = get_game_phase(board_before)

    tactics_before = analyze_tactical_patterns(board_before, turn)
    missed_tactic = tactics_before["total_tactics"] > 0

    king_info = analyze_king_safety(board_after, turn)
    king_exposed = king_info["safety_level"] != "safe"

    if missed_tactic:
        category = "missed_tactic"
        reason = f"A tactical opportunity was available but missed — {tactics_before['forks']} forks, {tactics_before['pins']} pins"
    elif king_exposed:
        category = "king_safety"
        reason = f"Move left the king exposed — {king_info['safety_level']}"
    elif phase == "opening":
        category = "opening_error"
        reason = "Poor opening move — likely a development or center control mistake"
    elif phase == "endgame":
        category = "endgame_error"
        reason = "Endgame technique error"
    else:
        category = "positional_error"
        reason = "General positional mistake — piece placement or structure"

    return {
        "classification": blunder_info["classification"],
        "drop": blunder_info["drop"],
        "category": category,
        "reason": reason,
        "phase": phase
    }


if __name__ == "__main__":
    board_before = chess.Board("r3k3/2N5/8/8/8/8/8/4K3 w - - 0 1")
    board_after  = chess.Board("r3k3/8/2N5/8/8/8/8/4K3 b - - 1 1")

    result = categorize_mistake(
        board_before=board_before,
        board_after=board_after,
        eval_before=2.0,
        eval_after=0.2,
        turn=chess.WHITE
    )

    print("Mistake categorization:")
    for key, value in result.items():
        print(f"  {key}: {value}")