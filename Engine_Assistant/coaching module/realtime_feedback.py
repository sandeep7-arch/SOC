# realtime_feedback.py

def generate_feedback(weakness_type):
    """
    Generates real-time coaching feedback.

    Parameters
    ----------
    weakness_type : str

    Returns
    -------
    dict
    """

    feedback_map = {

        "missed_tactic": {
            "severity": "warning",
            "message": "Look carefully for tactical opportunities before moving."
        },

        "king_safety": {
            "severity": "warning",
            "message": "Your king appears exposed. Consider improving king safety."
        },

        "endgame_error": {
            "severity": "info",
            "message": "Pay attention to endgame fundamentals."
        },

        "opening_error": {
            "severity": "info",
            "message": "Focus on development and center control."
        },

        "poor_center_control": {
            "severity": "warning",
            "message": "Try to increase control of the central squares."
        },

        "low_piece_activity": {
            "severity": "warning",
            "message": "Some pieces are inactive. Look for better placement."
        },

        "doubled_pawns": {
            "severity": "info",
            "message": "Your pawn structure contains doubled pawns."
        },

        "isolated_pawns": {
            "severity": "info",
            "message": "Watch out for isolated pawns becoming targets."
        }
    }

    return feedback_map.get(
        weakness_type,
        {
            "severity": "info",
            "message": "Continue evaluating the position carefully."
        }
    )


def generate_multiple_feedback(weaknesses):
    """
    Generates feedback for multiple weaknesses.

    Example:
    [
        "king_safety",
        "missed_tactic"
    ]
    """

    feedback_list = []

    for weakness in weaknesses:

        feedback = generate_feedback(
            weakness
        )

        feedback_list.append(
            {
                "weakness": weakness,
                "severity": feedback["severity"],
                "message": feedback["message"]
            }
        )

    return feedback_list


if __name__ == "__main__":

    weaknesses = [
        "king_safety",
        "missed_tactic",
        "poor_center_control"
    ]

    feedback = generate_multiple_feedback(
        weaknesses
    )

    for item in feedback:
        print(item)