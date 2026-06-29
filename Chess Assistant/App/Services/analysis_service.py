import sys
import os

# Get the Chess Assistant project root.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

sys.path.append(os.path.join(ROOT, 'Engine_Assistant', 'Analysis'))
sys.path.append(os.path.join(ROOT, 'Engine_Assistant', 'Player model'))  # space in folder name is fine with os.path.join
sys.path.append(os.path.join(ROOT, 'Database'))
sys.path.append(os.path.join(ROOT, 'Database', 'models'))
sys.path.append(ROOT)

import chess
from engine_service import get_move_and_score, get_position_score
from Engine_Assistant.Analysis.blunder_detector import blunder_detector
from Engine_Assistant.Analysis.game_phase import get_game_phase
from Engine_Assistant.Analysis.mistake_categorizer import categorize_mistake
from Engine_Assistant.Analysis.weakness_detector import detect_weaknesses
from Engine_Assistant.Analysis.tactical_patterns import analyze_tactical_patterns


def analyze_single_move(board_before: chess.Board, move_uci: str, depth: int = 8) -> dict:
    
    # Analyses a single move in real time.
    # Used by practice mode for instant feedback after each move.
    
    move = chess.Move.from_uci(move_uci)

    # Eval before
    fen_before = board_before.fen()
    eval_before = get_position_score(board_before.fen(), depth=depth)
    best_move, best_score = get_move_and_score(fen_before, depth=depth)
    phase = get_game_phase(board_before)
    turn = board_before.turn

    # Make the move
    board_after = board_before.copy()
    board_after.push(move)

    # Eval after
    eval_after = get_position_score(board_after.fen(), depth=depth)

    # Blunder detection
    blunder_info = blunder_detector(eval_before, eval_after, turn)
    category_info = categorize_mistake(
        board_before=board_before,
        board_after=board_after,
        eval_before=eval_before,
        eval_after=eval_after,
        turn=turn
    )
    tactics = analyze_tactical_patterns(board_before, turn)

    return {
        "move": move_uci,
        "fen_before": fen_before,
        "piece": chess.piece_name(board_before.piece_at(move.from_square).piece_type)
        if board_before.piece_at(move.from_square) else "piece",
        "phase": phase,
        "eval_before": eval_before,
        "eval_after": eval_after,
        "best_move": best_move,
        "best_score": best_score,
        "drop": blunder_info["drop"],
        "classification": blunder_info["classification"],
        "category": category_info.get("category"),
        "reason": category_info.get("reason"),
        "tactics_before": tactics
    }


def analyze_full_game(moves_list: list, player_color=None, depth: int = 8) -> dict:
    
    # Analyses a complete game after it ends.
    # Used by match mode for post-game report.
    # moves_list: list of UCI moves e.g. ["e2e4", "e7e5", ...]
    # player_color: chess.WHITE/chess.BLACK to count only the human player's mistakes.
    
    board = chess.Board()
    move_results = []

    for move_uci in moves_list:
        move = chess.Move.from_uci(move_uci)

        fen_before = board.fen()
        eval_before = get_position_score(fen_before, depth=depth)
        best_move, best_score = get_move_and_score(fen_before, depth=depth)
        phase = get_game_phase(board)
        turn = board.turn
        board_before = board.copy(stack=False)
        piece = board.piece_at(move.from_square)

        board.push(move)

        eval_after = get_position_score(board.fen(), depth=depth)
        blunder_info = blunder_detector(eval_before, eval_after, turn)
        category_info = categorize_mistake(
            board_before=board_before,
            board_after=board,
            eval_before=eval_before,
            eval_after=eval_after,
            turn=turn
        )

        move_results.append({
            "move_number": len(board.move_stack),
            "move": move_uci,
            "fen_before": fen_before,
            "piece": chess.piece_name(piece.piece_type) if piece else "piece",
            "played_by": "white" if turn == chess.WHITE else "black",
            "is_player_move": player_color is None or turn == player_color,
            "phase": phase,
            "eval_before": eval_before,
            "eval_after": eval_after,
            "best_move": best_move,
            "best_score": best_score,
            "drop": blunder_info["drop"],
            "classification": blunder_info["classification"],
            "category": category_info.get("category"),
            "reason": category_info.get("reason")
        })

    # Final board weaknesses
    weakness_color = player_color if player_color is not None else chess.WHITE
    final_weaknesses = detect_weaknesses(board, weakness_color)

    # Summary stats
    counted_moves = [m for m in move_results if m["is_player_move"]]
    blunders     = [m for m in counted_moves if m["classification"] == "blunder"]
    mistakes     = [m for m in counted_moves if m["classification"] == "mistake"]
    inaccuracies = [m for m in counted_moves if m["classification"] == "inaccuracy"]

    return {
        "moves": move_results,
        "player_moves": counted_moves,
        "total_blunders": len(blunders),
        "total_mistakes": len(mistakes),
        "total_inaccuracies": len(inaccuracies),
        "blunder_moves": blunders,
        "final_weaknesses": final_weaknesses,
        "total_moves": len(moves_list)
    }


if __name__ == "__main__":
    # Test with a short mock game
    test_moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d3", "f8c5"]

    print("Testing single move analysis:")
    board = chess.Board()
    result = analyze_single_move(board, "e2e4")
    print(f"  Move: {result['move']}")
    print(f"  Classification: {result['classification']}")
    print(f"  Drop: {result['drop']}")
    print()

    print("Testing full game analysis:")
    game = analyze_full_game(test_moves)
    print(f"  Total moves: {game['total_moves']}")
    print(f"  Blunders: {game['total_blunders']}")
    print(f"  Mistakes: {game['total_mistakes']}")
    print(f"  Inaccuracies: {game['total_inaccuracies']}")
    print()
    print("  Move by move:")
    for m in game["moves"]:
        print(f"    Move {m['move_number']}: {m['move']:<8} "
              f"{m['classification']:<12} drop={m['drop']}")
