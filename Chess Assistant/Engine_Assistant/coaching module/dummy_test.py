# dummy_test.py

from recommendation_engine import generate_recommendations
from curriculum_builder import build_curriculum
from realtime_feedback import generate_multiple_feedback
from postgame_report import build_postgame_report

# ---------------------------------------------------
# Dummy output from Analysis Module
# ---------------------------------------------------

analysis_output = {
    "color": "white",
    "weaknesses": [
        {
            "type": "king_safety",
            "detail": "King is dangerously exposed"
        },
        {
            "type": "poor_center_control",
            "detail": "Only controlling one center square"
        },
        {
            "type": "low_piece_activity",
            "detail": "Average piece activity is low"
        }
    ],
    "weakness_count": 3
}

# ---------------------------------------------------
# Convert analysis output into top weaknesses
# (Recommendation Engine expects (weakness, score))
# ---------------------------------------------------

top_weaknesses = []

score = 1.0

for item in analysis_output["weaknesses"]:
    top_weaknesses.append((item["type"], round(score, 2)))
    score -= 0.15

# Simple weakness names
weaknesses = [w for w, s in top_weaknesses]

print("\nDetected Weaknesses")
print("---------------------")
print(weaknesses)

# ---------------------------------------------------
# Recommendation Engine
# ---------------------------------------------------

recommendations = generate_recommendations(top_weaknesses)

print("\nRecommendations")
print("---------------------")

for r in recommendations:
    print(r)

# ---------------------------------------------------
# Curriculum
# ---------------------------------------------------

curriculum = build_curriculum(top_weaknesses)

print("\nCurriculum")
print("---------------------")

for lesson in curriculum:
    print(lesson)

# ---------------------------------------------------
# Realtime Feedback
# ---------------------------------------------------

feedback = generate_multiple_feedback(weaknesses)

print("\nRealtime Feedback")
print("---------------------")

for f in feedback:
    print(f)

# ---------------------------------------------------
# Dummy Personalization Context
# (Pretend this came from Player Model Module)
# ---------------------------------------------------

personalization_context = {
    "player_name": "Shrawani",
    "rating": 1200,
    "win_rate": 0.56,
    "playing_style": "Aggressive",

    "top_weaknesses": [
        ("king_safety", 1.0),
        ("poor_center_control", 0.85),
        ("low_piece_activity", 0.70)
    ]
}

# ---------------------------------------------------
# Post Game Report
# ---------------------------------------------------

report = build_postgame_report(
    analysis_output,
    personalization_context
)

from postgame_report import format_report

print("\nPost Game Report")
print("---------------------")
print(format_report(report))