import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import Column, Integer, Float, String, ForeignKey, DateTime, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from base import Base


class Recommendation(Base):
    # Stores coaching recommendations given to a player. Each row denotes one recommendation after one game. Tracks whether the player acted on it and if it helped

    __tablename__ = "recommendations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable= False)
    recorded_at = Column(DateTime, default= func.now())

    # Weakness targeted by the recommebdation
    weakness_category = Column(String)

    # The actual recommendation text either from LLM or hardcoded
    recommendation_text = Column(String)

    # Priority of the recommendation in between 0.0 to 1.0
    priority = Column(Float, default = 0.5)

    # Whether the player score improved in this category after this recommendation.
    # True = Imporoved, False = Didn't Improve, None = Not checked yet
    was_helpful = Column(Boolean, nullable=True)

    player = relationship("Player", back_populates = "recommendations")

    def __repr__(self):
        return(f"<Recommendation(category = {self.weakness_category},"
               f"priority = {self.priority},"
               f"helpful = {self.was_helpful})")

if __name__ == "__main__":
    from base import init_db, get_session
    from game import Game
    from mistake import Mistake
    from vulnerability_vector import VulnerabilityVector

    init_db()
    session = get_session()

    player = Player(name = "Jonathan", rating = 1200)
    session.add(player)
    session.commit()

    rec1 = Recommendation(
        player_id = player.id,
        weakness_category = "missed_tactic",
        recommendation_text = "Practice knight fork patterns - you've missed this in 4 of your last 6 games. Try puzzles on chess.com targeting forks",
        priority = 0.9
    )
    session.add(rec1)

    rec2 = Recommendation(
        player_id = player.id,
        weakness_category = "king_safety",
        recommendation_text = "Castle earlier - your king was exposed in the last 3 games. Aim to castle before move 10.",
        priority = 0.7
    )
    session.add(rec2)
    session.commit()

    print("Recommendation saved:")
    recs = session.query(Recommendation).filter_by(player_id = player.id).all()
    for r in recs:
        print(f"   {r}")
        print(f"   Text:{r.recommendation_text}")
        print()

    session.refresh(rec1)
    rec1.was_helpful = True
    session.commit()

    print("After marking rec1 as helfpul:")
    updated = session.query(Recommendation).filter_by(
        player_id = player.id,
        was_helpful = True
    ).all()
    print(f"   Helpful recommendations:  {updated}")

    session.commit()
    session.close()