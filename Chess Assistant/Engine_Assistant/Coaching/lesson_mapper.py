# lesson_mapper.py

"""
Maps chess weaknesses to training lessons.

Used by:
- recommendation_engine.py
- curriculum_builder.py
- postgame_report.py
"""

LESSON_MAP = {
    "missed_tactic": {
        "title": "Tactical Training",
        "description": "Practice tactical puzzles involving forks, pins, skewers, and discovered attacks."
    },

    "king_safety": {
        "title": "King Safety",
        "description": "Study castling, pawn shields, and defending against attacking pieces."
    },

    "positional_error": {
        "title": "Positional Play",
        "description": "Learn piece coordination, weak squares, and long-term strategic planning."
    },

    "endgame_error": {
        "title": "Endgame Fundamentals",
        "description": "Practice king and pawn endgames, opposition, and conversion techniques."
    },

    "opening_error": {
        "title": "Opening Principles",
        "description": "Improve development, center control, king safety, and opening preparation."
    },

    "doubled_pawns": {
        "title": "Pawn Structure",
        "description": "Learn how doubled pawns affect mobility, weaknesses, and planning."
    },

    "isolated_pawns": {
        "title": "Isolated Pawn Structures",
        "description": "Study strengths and weaknesses of isolated pawns and how to play around them."
    },

    "low_piece_activity": {
        "title": "Piece Activity",
        "description": "Improve piece placement, coordination, and active play."
    },

    "poor_center_control": {
        "title": "Center Control",
        "description": "Learn how to occupy and control central squares effectively."
    }
}


def get_lesson(weakness_type):
    """
    Returns lesson information for a weakness.
    """

    return LESSON_MAP.get(
        weakness_type,
        {
            "title": "General Improvement",
            "description": "Review recent games and focus on overall chess fundamentals."
        }
    )


def get_all_lessons():
    """
    Returns all available lessons.
    """

    return LESSON_MAP
