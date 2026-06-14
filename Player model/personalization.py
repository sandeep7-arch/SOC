from vulnerability_vector import build_vulnerability_vector, get_top_weaknesses
from style_classifier import classify_style


def build_personalization_context(profile, game_index):

    # The final output of player_model/.
    # Combines profile, vulnerability vector, style, and top weaknesses into one object ready to be passed to the LLM.

    vector = build_vulnerability_vector(game_index)
    top_weaknesses = get_top_weaknesses(vector, n=3)
    style_info = classify_style(vector, game_index)

    return {
        "player_name": profile.name,
        "rating": profile.rating,
        "total_games": profile.total_games,
        "win_rate": profile.win_rate(),
        "playing_style": style_info["style"],
        "style_confidence": style_info["confidence"],
        "vulnerability_vector": vector,
        "top_weaknesses": top_weaknesses
    }


def build_llm_context_string(context):

    # Converts the context into a readable text block that can be inserted directly into an LLM prompt.

    lines = []
    lines.append(f"Player: {context['player_name']} (Rating: {context['rating']})")
    lines.append(f"Games played: {context['total_games']}, Win rate: {context['win_rate']}")
    lines.append(f"Playing style: {context['playing_style']} (confidence: {context['style_confidence']})")
    lines.append("Top weaknesses:")

    for weakness, score in context["top_weaknesses"]:
        lines.append(f"  - {weakness}: {score}")

    return "\n".join(lines)


if __name__ == "__main__":
    from profile import player_profile
    from game_index import GameIndex, GameRecord

    # Build a profile
    profile = player_profile(name="Steve", rating=1200)
    profile.update_result("loss")
    profile.update_result("win")
    profile.update_result("loss")

    index = GameIndex()
    index.add_game(GameRecord(
        game_id="game_001", result="loss", opponent_rating=1400,
        phase_reached="middlegame", total_moves=35,
        blunder_count=2, mistake_count=3, inaccuracy_count=4,
        weaknesses_found=["missed_tactic", "king_safety", "missed_tactic"]
    ))
    index.add_game(GameRecord(
        game_id="game_002", result="win", opponent_rating=1150,
        phase_reached="endgame", total_moves=52,
        blunder_count=1, mistake_count=1, inaccuracy_count=2,
        weaknesses_found=["endgame_error", "missed_tactic"]
    ))
    index.add_game(GameRecord(
        game_id="game_003", result="loss", opponent_rating=1300,
        phase_reached="middlegame", total_moves=28,
        blunder_count=3, mistake_count=2, inaccuracy_count=1,
        weaknesses_found=["king_safety", "positional_error", "missed_tactic"]
    ))

    context = build_personalization_context(profile, index)

    print("Personalization context (raw):")
    print(context)
    print()

    print("LLM-ready context string:")
    print(build_llm_context_string(context))