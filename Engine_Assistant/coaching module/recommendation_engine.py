# recommendation_engine.py

from lesson_mapper import get_lesson


def generate_recommendations(top_weaknesses):
    """
    Generates training recommendations from top weaknesses.

    Parameters
    ----------
    top_weaknesses : list

    Example:
    [
        ("missed_tactic", 0.80),
        ("king_safety", 0.60),
        ("endgame_error", 0.40)
    ]

    Returns
    -------
    list[dict]
    """

    recommendations = []

    for weakness, score in top_weaknesses:

        lesson = get_lesson(weakness)

        recommendations.append({
            "weakness": weakness,
            "score": round(score, 2),
            "lesson_title": lesson["title"],
            "lesson_description": lesson["description"]
        })

    return recommendations


def get_priority_recommendation(top_weaknesses):
    """
    Returns the highest priority recommendation.
    """

    if not top_weaknesses:
        return None

    weakness, score = top_weaknesses[0]

    lesson = get_lesson(weakness)

    return {
        "weakness": weakness,
        "score": round(score, 2),
        "lesson_title": lesson["title"],
        "lesson_description": lesson["description"]
    }


def format_recommendations(recommendations):
    """
    Converts recommendation objects into readable text.
    Useful for Streamlit and LLM prompts.
    """

    lines = []

    for i, rec in enumerate(recommendations, start=1):

        lines.append(
            f"{i}. {rec['lesson_title']} "
            f"(weakness score: {rec['score']})"
        )

        lines.append(
            f"   {rec['lesson_description']}"
        )

    return "\n".join(lines)


if __name__ == "__main__":

    sample_top_weaknesses = [
        ("missed_tactic", 0.85),
        ("king_safety", 0.70),
        ("endgame_error", 0.40)
    ]

    recommendations = generate_recommendations(
        sample_top_weaknesses
    )

    print("Recommendations:\n")

    for rec in recommendations:
        print(rec)

    print("\nTop Priority Recommendation:\n")

    print(
        get_priority_recommendation(
            sample_top_weaknesses
        )
    )

    print("\nFormatted Output:\n")

    print(
        format_recommendations(
            recommendations
        )
    )