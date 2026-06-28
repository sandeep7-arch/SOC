# postgame_report.py

from recommendation_engine import generate_recommendations


def build_postgame_report(
    analysis_result,
    personalization_context
):
    """
    Builds a complete post-game coaching report.

    Parameters
    ----------
    analysis_result : dict

    Example:
    {
        "weaknesses": [
            {
                "type": "king_safety",
                "detail": "King is exposed"
            }
        ],
        "weakness_count": 1
    }

    personalization_context : dict

    Output of:
    build_personalization_context()

    Returns
    -------
    dict
    """

    top_weaknesses = personalization_context[
        "top_weaknesses"
    ]

    recommendations = generate_recommendations(
        top_weaknesses
    )

    report = {
        "player_name":
            personalization_context["player_name"],

        "rating":
            personalization_context["rating"],

        "playing_style":
            personalization_context["playing_style"],

        "win_rate":
            personalization_context["win_rate"],

        "weakness_count":
            analysis_result["weakness_count"],

        "detected_weaknesses":
            analysis_result["weaknesses"],

        "recommendations":
            recommendations
    }

    return report


def format_report(report):
    """
    Converts report dictionary
    into readable text.
    """

    lines = []

    lines.append("=" * 50)
    lines.append("POST GAME COACHING REPORT")
    lines.append("=" * 50)

    lines.append(
        f"Player: {report['player_name']}"
    )

    lines.append(
        f"Rating: {report['rating']}"
    )

    lines.append(
        f"Style: {report['playing_style']}"
    )

    lines.append(
        f"Win Rate: {report['win_rate']}"
    )

    lines.append("")

    lines.append(
        f"Weaknesses Found: {report['weakness_count']}"
    )

    lines.append("")

    for weakness in report[
        "detected_weaknesses"
    ]:

        lines.append(
            f"- {weakness['type']}: "
            f"{weakness['detail']}"
        )

    lines.append("")
    lines.append("Recommended Training")
    lines.append("-" * 25)

    for rec in report["recommendations"]:

        lines.append(
            f"* {rec['lesson_title']}"
        )

        lines.append(
            f"  {rec['lesson_description']}"
        )

    return "\n".join(lines)


if __name__ == "__main__":

    sample_analysis = {
        "weakness_count": 2,
        "weaknesses": [
            {
                "type": "king_safety",
                "detail":
                    "King is exposed"
            },
            {
                "type":
                    "poor_center_control",
                "detail":
                    "Only controlling 1 center square"
            }
        ]
    }

    sample_context = {
        "player_name": "Steve",
        "rating": 1200,
        "win_rate": 0.45,
        "playing_style": "Aggressive",

        "top_weaknesses": [
            ("king_safety", 0.8),
            ("missed_tactic", 0.7),
            ("endgame_error", 0.4)
        ]
    }

    report = build_postgame_report(
        sample_analysis,
        sample_context
    )

    print(
        format_report(report)
    )