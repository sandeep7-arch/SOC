import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

sys.path.append(ROOT)

from engine.explain.blunder_explainer import BlunderExplainer
from engine.explain.commentary import CommentaryEngine, CommentaryStyle
from engine.explain.lesson_extractor import LessonExtractor
from engine.explain.llm_client import get_llm
from engine.explain.move_explainer import MoveExplainer
from engine.explain.prompt_builder import MoveData


def pawns_to_centipawns(score: float) -> int:
    return int(round(score * 100))


def analysis_move_to_move_data(move: dict) -> MoveData:
    return MoveData(
        move_played=move["move"],
        score_before=pawns_to_centipawns(move["eval_before"]),
        score_after=pawns_to_centipawns(move["eval_after"]),
        best_move=move.get("best_move") or move["move"],
        best_score=pawns_to_centipawns(move.get("best_score", move["eval_after"])),
        category=move["classification"],
        piece=move.get("piece", "piece"),
        phase=move.get("phase", "middlegame"),
        move_number=move.get("move_number", 0),
        player_color=move.get("played_by", "white").capitalize(),
        fen_before=move.get("fen_before"),
        tactic_type=move.get("category"),
        reason=move.get("reason"),
    )


def explain_single_move(feedback: dict, provider: str = "mock", player_level: str = "intermediate") -> dict:
    move_data = analysis_move_to_move_data({
        "move_number": feedback.get("move_number", 0),
        "played_by": feedback.get("played_by", "white"),
        **feedback,
    })
    llm = get_llm(provider)
    explanation = MoveExplainer(llm=llm, player_level=player_level).explain_move(move_data)
    commentary = CommentaryEngine(
        llm=llm,
        style=CommentaryStyle.EDUCATIONAL,
        use_llm=False,
    ).comment(move_data)

    return {
        "move_data": move_data,
        "explanation": explanation,
        "commentary": commentary,
    }


def explain_game_analysis(analysis: dict, provider: str = "mock", player_level: str = "intermediate") -> dict:
    player_moves = analysis.get("player_moves", [])
    move_data = [analysis_move_to_move_data(move) for move in player_moves]
    bad_moves = [
        move for move in move_data
        if move.category.lower() in {"blunder", "mistake", "inaccuracy"}
    ]

    llm = get_llm(provider)
    blunder_report = BlunderExplainer(
        llm=llm,
        player_level=player_level,
    ).analyze_blunders(bad_moves)
    lesson_report = LessonExtractor(llm=llm).extract(
        bad_moves,
        player_color=move_data[0].player_color if move_data else "White",
    )

    return {
        "move_data": move_data,
        "bad_moves": bad_moves,
        "blunder_report": blunder_report,
        "lesson_report": lesson_report,
    }
