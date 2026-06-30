# recommendation_engine.py

from lesson_mapper import get_lesson


def generate_recommendations(top_weaknesses):
    """
    Generates training recommendations from top weaknesses.
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
