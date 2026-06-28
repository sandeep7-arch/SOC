# curriculum_builder.py

from lesson_mapper import get_lesson


def build_curriculum(top_weaknesses):
    """
    Builds a personalized training curriculum.

    Parameters
    ----------
    top_weaknesses : list

    Example:
    [
        ("missed_tactic", 0.85),
        ("king_safety", 0.70),
        ("endgame_error", 0.40)
    ]

    Returns
    -------
    dict
    """

    curriculum = {}

    week_number = 1

    for weakness, score in top_weaknesses:

        lesson = get_lesson(weakness)

        curriculum[f"Week {week_number}"] = {
            "focus_area": lesson["title"],
            "weakness": weakness,
            "score": score,
            "description": lesson["description"]
        }

        week_number += 1

    return curriculum


def build_detailed_curriculum(top_weaknesses):
    """
    Creates a more detailed study schedule.
    """

    curriculum = {}

    week_number = 1

    for weakness, score in top_weaknesses:

        lesson = get_lesson(weakness)

        curriculum[f"Week {week_number}"] = {
            "focus_area": lesson["title"],
            "weakness": weakness,
            "score": round(score, 2),

            "daily_plan": [
                "Study concepts",
                "Solve 10 practice positions",
                "Review master games",
                "Play training games",
                "Analyze mistakes"
            ],

            "description": lesson["description"]
        }

        week_number += 1

    return curriculum


def format_curriculum(curriculum):
    """
    Converts curriculum into readable text.
    """

    lines = []

    lines.append("=" * 50)
    lines.append("PERSONALIZED TRAINING CURRICULUM")
    lines.append("=" * 50)

    for week, info in curriculum.items():

        lines.append("")
        lines.append(f"{week}")
        lines.append("-" * 20)

        lines.append(
            f"Focus Area: {info['focus_area']}"
        )

        lines.append(
            f"Weakness: {info['weakness']}"
        )

        lines.append(
            f"Score: {info['score']}"
        )

        lines.append(
            f"Description: {info['description']}"
        )

        if "daily_plan" in info:

            lines.append("Daily Tasks:")

            for task in info["daily_plan"]:
                lines.append(f"  - {task}")

    return "\n".join(lines)


if __name__ == "__main__":

    sample_top_weaknesses = [
        ("missed_tactic", 0.85),
        ("king_safety", 0.70),
        ("endgame_error", 0.40)
    ]

    curriculum = build_detailed_curriculum(
        sample_top_weaknesses
    )

    print(
        format_curriculum(curriculum)
    )