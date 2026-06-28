# adaptive_engine.py

def determine_training_level(score):
    """
    Determines training difficulty based on weakness score.

    Parameters
    ----------
    score : float

    Returns
    -------
    str
    """

    if score >= 0.75:
        return "Beginner"

    elif score >= 0.40:
        return "Intermediate"

    return "Advanced"


def build_adaptive_plan(vulnerability_vector):
    """
    Creates personalized training priorities.

    Parameters
    ----------
    vulnerability_vector : dict

    Example:
    {
        "missed_tactic": 0.8,
        "king_safety": 0.6,
        "endgame_error": 0.3
    }

    Returns
    -------
    dict
    """

    plan = {}

    for weakness, score in vulnerability_vector.items():

        level = determine_training_level(score)

        plan[weakness] = {
            "score": round(score, 2),
            "training_level": level
        }

    return plan


def get_high_priority_areas(vulnerability_vector,
                            threshold=0.60):
    """
    Returns weaknesses requiring urgent attention.
    """

    priority = []

    for weakness, score in vulnerability_vector.items():

        if score >= threshold:

            priority.append({
                "weakness": weakness,
                "score": round(score, 2)
            })

    priority.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return priority


def build_adaptive_summary(vulnerability_vector):
    """
    Creates a readable coaching summary.
    """

    priority = get_high_priority_areas(
        vulnerability_vector
    )

    lines = []

    lines.append("Adaptive Training Summary")
    lines.append("-" * 30)

    if not priority:
        lines.append(
            "No major weaknesses detected."
        )

    else:

        for item in priority:

            lines.append(
                f"{item['weakness']} "
                f"(score={item['score']})"
            )

    return "\n".join(lines)


if __name__ == "__main__":

    sample_vector = {

        "missed_tactic": 0.90,

        "king_safety": 0.70,

        "endgame_error": 0.35,

        "opening_error": 0.20,

        "poor_center_control": 0.65
    }

    print("Adaptive Plan:\n")

    plan = build_adaptive_plan(
        sample_vector
    )

    for weakness, info in plan.items():

        print(
            weakness,
            "->",
            info
        )

    print("\n")

    print(
        build_adaptive_summary(
            sample_vector
        )
    )