import chess
import chess.engine
from game_phase import get_game_phase
from blunder_detector import blunder_detector

def classify_game(game_moves):
    board = chess.Board()
    results = []

    for move_data in game_moves:
        blunder_info = blunder_detector(eval_before = move_data["eval_before"],eval_after=move_data["eval_after"], turn=move_data["turn"])
        game_phase = get_game_phase(board)
        results.append({
            "move_number" : len(board.move_stack)+1,
            "move" : move_data["move"],
            "game_phase": game_phase,
            "eval_before": blunder_info["evaluation_before"],
            "eval_after": blunder_info["evaluation_after"],
            "drop": blunder_info["drop"],
            "classification": blunder_info["classification"]
        })

        board.push_san(move_data["move"])

    return results


def classify_game_from_engine(moves_list, engine_path, depth = 8):

    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    board = chess.Board()
    results
    
    for move_uci in moves_list:
        move = chess.Move.from_uci(move_uci)

        # Evaluation before the move
        info_before = engine.analysis(board,chess.engine.Limit(depth = depth))
        eval_before = info_before["score"].relative.score(mate_score = 10000)/100

        game_phase = get_game_phase(board)
        turn = board.turn

        board.push(move)

        #Evlauation after the move
        info_after = engine.analyse(board, chess.engine.Limit(depth = depth))
        eval_after = info_after["score"].relative.score(mate_score = 10000)/100

        blunder_info = blunder_detector(eval_before, eval_after, turn)
        
        results.append({
            "move_number": len(board.move_stack),
            "move" : move_uci,
            "game_phase" : game_phase,
            "eval_before" : eval_before,
            "eval_after" : eval_after,
            "drop" : blunder_info["drop"],
            "classification" : blunder_info["classification"]
        })

    engine.quit()
    return results

if __name__ == "__main__":
    game_moves = [
        {"move": "e4",  "eval_before": 0.0, "eval_after": 0.3,  "turn": chess.WHITE},
        {"move": "e5",  "eval_before": 0.3, "eval_after": 0.1,  "turn": chess.BLACK},
        {"move": "Nf3", "eval_before": 0.1, "eval_after": 0.4,  "turn": chess.WHITE},
        {"move": "Nc6", "eval_before": 0.4, "eval_after": 0.2,  "turn": chess.BLACK},
        {"move": "Bc4", "eval_before": 0.2, "eval_after": 0.5,  "turn": chess.WHITE},
        {"move": "d6",  "eval_before": 0.5, "eval_after": 1.8,  "turn": chess.BLACK},
    ]

    results = classify_game(game_moves)

    for r in results:
        print(f"Move {r['move_number']}: {r['move']:<6} | "
              f"Phase: {r['game_phase']:<12} | "
              f"Drop: {r['drop']:<6} | "
              f"{r['classification']}")