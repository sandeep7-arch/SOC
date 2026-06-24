import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import Column, Integer, String, Float, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from base import Base
from player import Player
from game import Game
from mistake import Mistake
from vulnerability_vector import VulnerabilityVector
from recommendation import Recommendation

class EnginePreferences(Base):
    # Stores each player's app setting's permanently. Each row denotes player's preferences between sessions.
    __tablename__ = "engine_preferences"

    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False, unique=True)

    # Engine strength settings
    engine_depth = Column(Integer, default=8)
    engine_strength = Column(Integer, default=1200)

    # Practice mode toggles
    hints_enabled = Column(Boolean, default=True)
    blunder_alerts_enabled = Column(Boolean, default=True)
    show_eval_bar = Column(Boolean, default=True)
    show_best_move = Column(Boolean, default= False) # off by default, spoils the game

    # Game mode - Practice or Match
    default_mode = Column(String, default="practice")

    # LLM coaching settings
    coaching_enabled = Column(Boolean, default=True)
    coaching_detail_level = Column(String, default= "preferences")

    player = relationship("Player", back_populates="preferences")

    def __repr__(self):
        return(f"<EnginePreferences(player_id = {self.player}"
               f"depth={self.engine_depth},"
               f"hints={self.hints_enabled},"
                f"mode = {self.default_mode})>")
    
if __name__ == "__main__":
    from base import init_db, get_session
    from player import Player
    from game import Game
    from mistake import Mistake
    from vulnerability_vector import VulnerabilityVector
    from recommendation import Recommendation

    init_db()
    session = get_session()

    # Create a player
    player = Player(name = "Jonathan", rating = 1200)
    session.add(player)
    session.commit()

    # Create default preferences for this player
    prefs = EnginePreferences(
        player_id = player.id,
        engine_depth = 10,
        hints_enabled = True,
        blunder_alerts_enabled = True,
        show_eval_bar = True,
        default_mode = "practice",
        coaching_enabled = True,
        coaching_detail_level = "practice"
    )
    session.add(prefs)
    session.commit()

    print("Preferences saved:", prefs)

    fetched = session.query(EnginePreferences).filter_by(
        player_id=player.id,
    ).first()
    print(f"Hints enabled: {fetched.hints_enabled}")
    print(f"Engine depth: {fetched.engine_depth}")
    print(f"Coaching detail: {fetched.coaching_detail_level}")

    session.refresh(fetched)
    fetched.hints_enabled = False
    session.commit()

    updated = session.query(EnginePreferences).filter_by(
        player_id = player.id,
    ).first()
    print(f"Hints after update: {updated.hints_enabled}")

    session.commit()
    session.close()