import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

sys.path.insert(0, os.path.join(ROOT, "Engine_Assistant", "Coaching"))
sys.path.insert(0, os.path.join(ROOT, "Engine_Assistant", "Player_model"))

from Engine_Assistant.Coaching.postgame_report import build_postgame_report, format_report
from Engine_Assistant.Coaching.recommendation_engine import generate_recommendations
from Engine_Assistant.Coaching.realtime_feedback import generate_multiple_feedback
from Engine_Assistant.Player_model.vulnerability_vector import ALL_WEAKNESS_TYPES, get_top_weaknesses


def collect_game_weaknesses(game_analysis: dict) -> list[str]:
    """
    Extracts weakness labels from the player's analyzed moves and final position.
    """
    weaknesses = []

    for move in game_analysis.get("player_moves", []):
        if move.get("classification") == "good":
            continue
        category = move.get("category")
        if category:
            weaknesses.append(category)

    for weakness in game_analysis.get("final_weaknesses", {}).get("weaknesses", []):
        weakness_type = weakness.get("type")
        if weakness_type:
            weaknesses.append(weakness_type)

    return weaknesses


def build_game_vulnerability_vector(game_analysis: dict) -> dict:
    """
    Builds a 0..1 vulnerability vector for the current game.
    Higher scores mean this weakness appeared more often in this game.
    """
    counts = {weakness_type: 0 for weakness_type in ALL_WEAKNESS_TYPES}

    for weakness in collect_game_weaknesses(game_analysis):
        if weakness in counts:
            counts[weakness] += 1

    max_count = max(counts.values(), default=0)
    if max_count == 0:
        return {weakness_type: 0.0 for weakness_type in ALL_WEAKNESS_TYPES}

    return {
        weakness_type: round(count / max_count, 2)
        for weakness_type, count in counts.items()
    }


def build_personalized_context(player_profile: dict, game_analysis: dict) -> dict:
    """
    Combines stored player data with the newest game so recommendations are player-specific.
    """
    current_vector = build_game_vulnerability_vector(game_analysis)
    stored_vector = player_profile.get("vulnerability_vector") or {}

    merged_vector = {}
    for weakness_type in ALL_WEAKNESS_TYPES:
        stored_score = float(stored_vector.get(weakness_type, 0.0) or 0.0)
        current_score = float(current_vector.get(weakness_type, 0.0) or 0.0)
        merged_vector[weakness_type] = round(max(stored_score, current_score), 2)

    top_weaknesses = [
        item for item in get_top_weaknesses(merged_vector, n=3)
        if item[1] > 0
    ]

    if not top_weaknesses:
        top_weaknesses = [("positional_error", 0.1)]

    return {
        "player_name": player_profile.get("name", "Player"),
        "rating": player_profile.get("rating", 1200),
        "total_games": player_profile.get("total_games", 0),
        "win_rate": player_profile.get("win_rate", 0.0),
        "playing_style": player_profile.get("playing_style") or "unknown",
        "vulnerability_vector": merged_vector,
        "top_weaknesses": top_weaknesses,
    }


def build_coaching_report(player_profile: dict, game_analysis: dict) -> dict:
    context = build_personalized_context(player_profile, game_analysis)
    analysis_result = game_analysis.get("final_weaknesses", {})

    if not analysis_result.get("weaknesses"):
        analysis_result = {
            "weakness_count": len(collect_game_weaknesses(game_analysis)),
            "weaknesses": [
                {"type": weakness, "detail": "Seen in your move choices this game."}
                for weakness, _score in context["top_weaknesses"]
            ],
        }

    report = build_postgame_report(analysis_result, context)
    report["realtime_feedback"] = generate_multiple_feedback(
        [weakness for weakness, _score in context["top_weaknesses"]]
    )
    return report


def format_coaching_report(report: dict) -> str:
    return format_report(report)


def recommendations_from_analysis(player_profile: dict, game_analysis: dict) -> list[dict]:
    context = build_personalized_context(player_profile, game_analysis)
    return generate_recommendations(context["top_weaknesses"])
