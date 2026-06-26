import argparse
import os
import sys

import chess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.path.append(os.path.join(PROJECT_ROOT, "App", "Services"))
sys.path.append(os.path.join(PROJECT_ROOT, "Database"))

from analysis_service import analyze_full_game, analyze_single_move
from engine_service import close_stockfish, get_best_move, get_position_score
from explain_service import explain_game_analysis, explain_single_move
from profile_service import get_or_create_player, save_game_to_db
from base import init_db
from models import EnginePreferences, Game, Mistake, Player, Recommendation, VulnerabilityVector


def parse_human_move(board: chess.Board, move_text: str) -> chess.Move:
    move_text = move_text.strip()

    try:
        move = chess.Move.from_uci(move_text)
        if move in board.legal_moves:
            return move
    except ValueError:
        pass

    try:
        return board.parse_san(move_text)
    except ValueError as exc:
        raise ValueError("Enter a legal move in SAN like Nf3 or UCI like g1f3.") from exc


def result_for_player(board: chess.Board, player_color: chess.Color, resigned: bool = False) -> str:
    if resigned:
        return "loss"

    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return "draw"
    return "win" if outcome.winner == player_color else "loss"


def print_position(board: chess.Board):
    print()
    print(board)
    print(f"FEN: {board.fen()}")
    print(f"Eval: {get_position_score(board.fen()):+.2f}")
    print()


def play_game(player_name: str, player_color: chess.Color, depth: int, save: bool, llm_provider: str):
    init_db()
    player = get_or_create_player(player_name)
    player_id = player.id
    saved_player_name = player.name

    board = chess.Board()
    moves = []
    resigned = False

    print(f"Playing as {'White' if player_color == chess.WHITE else 'Black'} against Stockfish.")
    print("Type a move as SAN/UCI, or type 'resign' / 'quit'.")

    try:
        while not board.is_game_over(claim_draw=True):
            print_position(board)

            if board.turn == player_color:
                raw_move = input("Your move: ").strip()
                if raw_move.lower() in {"quit", "exit"}:
                    print("Game stopped without saving.")
                    return
                if raw_move.lower() == "resign":
                    resigned = True
                    break

                try:
                    move = parse_human_move(board, raw_move)
                except ValueError as exc:
                    print(exc)
                    continue

                feedback = analyze_single_move(board, move.uci(), depth=depth)
                feedback["move_number"] = len(board.move_stack) + 1
                feedback["played_by"] = "white" if board.turn == chess.WHITE else "black"
                board.push(move)
                moves.append(move.uci())

                print(
                    f"Feedback: {feedback['classification']} "
                    f"(drop {feedback['drop']:+.2f}, "
                    f"eval {feedback['eval_before']:+.2f} -> {feedback['eval_after']:+.2f})"
                )
                if feedback["category"]:
                    print(f"Category: {feedback['category']}")
                    print(f"Reason: {feedback['reason']}")

                tactics = feedback["tactics_before"]
                if tactics["total_tactics"]:
                    print(
                        "Tactics available before move: "
                        f"{len(tactics['forks'])} forks, "
                        f"{len(tactics['pins'])} pins, "
                        f"{len(tactics['skewers'])} skewers"
                    )

                explained = explain_single_move(feedback, provider=llm_provider)
                print(f"Coach: {explained['commentary'].text}")
                print(f"Explanation: {explained['explanation'].explanation}")
            else:
                engine_move_uci = get_best_move(board.fen(), depth=depth)
                engine_move = chess.Move.from_uci(engine_move_uci)
                san = board.san(engine_move)
                board.push(engine_move)
                moves.append(engine_move_uci)
                print(f"Stockfish plays: {san} ({engine_move_uci})")

        analysis = analyze_full_game(moves, player_color=player_color, depth=depth)
        result = result_for_player(board, player_color, resigned=resigned)

        print()
        print("Game over.")
        print(f"Result for {saved_player_name}: {result}")
        print(f"Your blunders: {analysis['total_blunders']}")
        print(f"Your mistakes: {analysis['total_mistakes']}")
        print(f"Your inaccuracies: {analysis['total_inaccuracies']}")

        for move in analysis["player_moves"]:
            if move["classification"] != "good":
                print(
                    f"Move {move['move_number']} {move['move']}: "
                    f"{move['classification']} in {move['phase']} "
                    f"(drop {move['drop']:+.2f})"
                )
                if move["category"]:
                    print(f"  Category: {move['category']}")
                if move["reason"]:
                    print(f"  Reason: {move['reason']}")

        final_weaknesses = analysis["final_weaknesses"]
        if final_weaknesses["weaknesses"]:
            print()
            print("Final position weaknesses:")
            for weakness in final_weaknesses["weaknesses"]:
                print(f"- {weakness['type']}: {weakness['detail']}")
        else:
            print()
            print("Final position weaknesses: none detected.")

        explained_game = explain_game_analysis(analysis, provider=llm_provider)
        print()
        print("Coaching pattern:")
        print(explained_game["blunder_report"].pattern_summary)

        if explained_game["lesson_report"].priority_lesson:
            lesson = explained_game["lesson_report"].priority_lesson
            print()
            print("Priority lesson:")
            print(f"{lesson.title}: {lesson.action_item}")

        if save:
            game_id = save_game_to_db(
                player_id,
                analysis,
                result=result,
                opponent_rating=analysis.get("opponent_rating", 3200),
            )
            print(f"Saved game to database with id {game_id}.")
    finally:
        close_stockfish()


def main():
    parser = argparse.ArgumentParser(description="Play a terminal game against Stockfish.")
    parser.add_argument("--player", default="Piyush", help="Player name to load/save in the database.")
    parser.add_argument("--color", choices=["white", "black"], default="white")
    parser.add_argument("--depth", type=int, default=8, help="Stockfish search depth.")
    parser.add_argument("--no-save", action="store_true", help="Analyze but do not save the game.")
    parser.add_argument(
        "--llm-provider",
        default="mock",
        choices=["mock", "gemini", "claude", "openai", "ollama"],
        help="Explanation backend. Mock works offline.",
    )
    args = parser.parse_args()

    player_color = chess.WHITE if args.color == "white" else chess.BLACK
    play_game(args.player, player_color, args.depth, save=not args.no_save, llm_provider=args.llm_provider)


if __name__ == "__main__":
    main()
