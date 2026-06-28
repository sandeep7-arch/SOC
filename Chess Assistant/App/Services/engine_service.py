import chess
import chess.engine
import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(ROOT, 'Engine_Assistant', 'Analysis'))
sys.path.append(os.path.join(ROOT, 'Engine_Assistant', 'Player model'))
sys.path.append(os.path.join(ROOT, 'Database'))
sys.path.append(os.path.join(ROOT, 'Database', 'models'))
sys.path.append(ROOT)
import chess

USE_STOCKFISH = True
USE_MOCK = False

STOCKFISH_FILENAME = "stockfish-windows-x86-64-avx2.exe"


def find_stockfish_path() -> str:
    candidates = [
        os.path.join(ROOT, "stockfish", STOCKFISH_FILENAME),
        os.path.join(os.path.dirname(ROOT), "stockfish", STOCKFISH_FILENAME),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    return candidates[0]


STOCKFISH_PATH = find_stockfish_path()


# Real engine config — update paths when team lead provides files
DLL_PATH   = os.path.join(ROOT, "Engine_Assistant", "search", "native_engine.dll")
MODEL_PATH = os.path.join(ROOT, "exports", "nnue_weights.bin")

_stockfish_instance = None

def get_stockfish():
    global _stockfish_instance
    if _stockfish_instance is None:
        if not os.path.exists(STOCKFISH_PATH):
            raise FileNotFoundError(f"Stockfish executable not found: {STOCKFISH_PATH}")
        _stockfish_instance = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    return _stockfish_instance


def close_stockfish():
    global _stockfish_instance
    if _stockfish_instance is not None:
        _stockfish_instance.quit()
        _stockfish_instance = None

def get_best_move(fen: str, depth: int = 8) -> str:
    # Returns best move for a position in UCI format e.g. 'e2e4'. Uses mock if engine not available.
    
    if USE_MOCK:
        # Return a random legal move as mock
        board = chess.Board(fen)
        legal = list(board.legal_moves)
        return legal[0].uci() if legal else "0000"

    if USE_STOCKFISH:
        engine = get_stockfish()
        board = chess.Board(fen)
        result = engine.play(board, chess.engine.Limit(depth=depth))
        return result.move.uci()
    
    engine = get_engine()
    move, _ = engine.get_best_move_with_score(fen, depth, 1000.0)
    return move


def get_position_score(fen: str, depth: int = 8) -> float:
    # Returns centipawn score for a position, converted to pawns.
    # Positive = white is better, negative = black is better.
    # Uses mock if engine not available.
    
    if USE_MOCK:
        # Return a small random score as mock
        import random
        return round(random.uniform(-0.5, 0.5), 2)

    if USE_STOCKFISH:
        engine = get_stockfish()
        board = chess.Board(fen)
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        score = info["score"].white().score(mate_score=10000)
        return score / 100 if score is not None else 0.0
    
    engine = get_engine()
    _, score_cp = engine.get_best_move_with_score(fen, depth, 1000.0)
    return score_cp / 100  # convert centipawns to pawns


def get_move_and_score(fen: str, depth: int = 8) -> tuple:
    # Returns (best_move, score_in_pawns) for a position.
    # Most efficient — gets both in one engine call.
    
    if USE_MOCK:
        return get_best_move(fen, depth), get_position_score(fen, depth)

    if USE_STOCKFISH:
        return get_best_move(fen, depth), get_position_score(fen, depth)
    
    engine = get_engine()
    move, score_cp = engine.get_best_move_with_score(fen, depth, 1000.0)
    return move, score_cp / 100


if __name__ == "__main__":
    import chess.engine
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    board = chess.Board()
    info = engine.analyse(board, chess.engine.Limit(depth=10))
    print(info["score"])
    engine.quit()
