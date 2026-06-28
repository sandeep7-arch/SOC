# intervention_manager.py

def should_intervene(
    blunder_count,
    mistake_count,
    weakness_count
):
    """
    Determines whether coaching intervention
    is needed.

    Returns
    -------
    bool
    """

    if blunder_count >= 3:
        return True

    if mistake_count >= 5:
        return True

    if weakness_count >= 4:
        return True

    return False


def get_intervention_reason(
    blunder_count,
    mistake_count,
    weakness_count
):
    """
    Returns reason for intervention.
    """

    reasons = []

    if blunder_count >= 3:
        reasons.append(
            "Too many blunders detected"
        )

    if mistake_count >= 5:
        reasons.append(
            "High mistake count"
        )

    if weakness_count >= 4:
        reasons.append(
            "Multiple weaknesses found"
        )

    return reasons


def create_intervention(
    blunder_count,
    mistake_count,
    weakness_count
):
    """
    Creates intervention object.
    """

    intervention_needed = should_intervene(
        blunder_count,
        mistake_count,
        weakness_count
    )

    return {
        "intervention_required":
            intervention_needed,

        "reasons":
            get_intervention_reason(
                blunder_count,
                mistake_count,
                weakness_count
            )
    }


def get_training_suggestion(
    weakness_type
):
    """
    Quick suggestion for a weakness.
    """

    suggestions = {

        "missed_tactic":
            "Spend 15 minutes on tactical puzzles.",

        "king_safety":
            "Review castling and king protection.",

        "endgame_error":
            "Practice basic king and pawn endgames.",

        "opening_error":
            "Review opening principles.",

        "poor_center_control":
            "Study center occupation strategies.",

        "low_piece_activity":
            "Improve piece development."
    }

    return suggestions.get(
        weakness_type,
        "Review recent games carefully."
    )


if __name__ == "__main__":

    intervention = create_intervention(
        blunder_count=4,
        mistake_count=2,
        weakness_count=3
    )

    print(intervention)

    print()

    print(
        get_training_suggestion(
            "king_safety"
        )
    )