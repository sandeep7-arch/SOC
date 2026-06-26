import chess
from pawn_structure import analyze_pawn_structure
from king_safety import analyze_king_safety
from positional_analysis import analyze_positional

def detect_weaknesses(board, color):

    pawn_info       = analyze_pawn_structure(board, color)
    king_info       = analyze_king_safety(board, color)
    positional_info = analyze_positional(board, color)

    weaknesses = []

    if pawn_info["doubled_pawns"]:
        weaknesses.append({
            "type": "doubled_pawns",
            "detail": f"Doubled pawns on files: {pawn_info['doubled_pawns']}"
        })

    if pawn_info["isolated_pawns"]:
        weaknesses.append({
            "type": "isolated_pawns",
            "detail": f"Isolated pawns on files: {pawn_info['isolated_pawns']}"
        })

    if king_info["safety_level"] == "dangerous":
        weaknesses.append({
            "type": "king_safety",
            "detail": f"King is dangerously exposed, {king_info['attacker_count']} attackers nearby"
        })
    elif king_info["safety_level"] == "slightly exposed":
        weaknesses.append({
            "type": "king_safety",
            "detail": f"King is slightly exposed with {king_info['pawn_shield']} pawn shield"
        })

    if positional_info["center_control"] <= 1:
        weaknesses.append({
            "type": "poor_center_control",
            "detail": f"Only controlling {positional_info['center_control']} center squares"
        })

    if positional_info["piece_activity"] < 3.0:
        weaknesses.append({
            "type": "low_piece_activity",
            "detail": f"Average piece activity is low: {positional_info['piece_activity']}"
        })

    return {
        "color": "white" if color == chess.WHITE else "black",
        "weaknesses": weaknesses,
        "weakness_count": len(weaknesses),
        "pawn_details": pawn_info,
        "king_details": king_info,
        "positional_details": positional_info
    }


if __name__ == "__main__":
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Nf3")
    board.push_san("Nc6")

    result = detect_weaknesses(board, chess.WHITE)
    print("White weaknesses:")
    for w in result["weaknesses"]:
        print(f"  - {w['type']}: {w['detail']}")
    print(f"  Total: {result['weakness_count']}")
    print()

    result = detect_weaknesses(board, chess.BLACK)
    print("Black weaknesses:")
    for w in result["weaknesses"]:
        print(f"  - {w['type']}: {w['detail']}")
    print(f"  Total: {result['weakness_count']}")

    board2 = chess.Board("r1bqk2r/pp3ppp/2n1pn2/3p4/2PP4/2PB4/P4PPP/RNBQK2R w KQkq - 0 1")
    print("\nPosition with weaknesses:")
    print(board2)
    print()

    result = detect_weaknesses(board2, chess.WHITE)
    print("White weaknesses:")
    for w in result["weaknesses"]:
        print(f"  - {w['type']}: {w['detail']}")
    print(f"  Total: {result['weakness_count']}")
    print()

    result = detect_weaknesses(board2, chess.BLACK)
    print("Black weaknesses:")
    for w in result["weaknesses"]:
        print(f"  - {w['type']}: {w['detail']}")
    print(f"  Total: {result['weakness_count']}")