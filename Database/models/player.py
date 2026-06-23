import sys
import os

# Add parent folder (database/) to path so we can import base.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.orm import relationship
from base import Base

class Player(Base):
    # Database table for storing player information permanently. Similar to PlayerProfile but for storage
    __tablename__ = "players"

    id = Column(Integer, primary_key=True, autoincrement=True)

    name = Column(String, nullable= False)
    rating = Column(Integer, default=1200)

    total_games = Column(Integer, default=0)
    wins = Column(Integer, default = 0)
    losses = Column(Integer, default=0)
    draws = Column(Integer, default =0)

    playing_style = Column(String, default="unknown")

    def win_rate(self) :
        if self.total_games == 0:
            return 0.0
        return round(self.wins/ self.total_games, 2)
    
    def __repr__(self):
        return f"<Player(name = '{self.name}', rating = {self.rating}, games = {self.total_games})>"
    
    games = relationship("Game", back_populates="player")
    mistakes = relationship("Mistake", back_populates="player")
    vulnerability_vectors = relationship("VulnerabilityVector", back_populates="player")
    recommendations = relationship("Recommendation", back_populates="player")
    
if __name__ == "__main__":

    from base import init_db, get_session

    init_db()

    session = get_session()

    new_player = Player(name = "Jonathan", rating = 1200)
    session.add(new_player)
    session.commit()

    print("Player saved:", new_player)
    print("Player ID assigned:", new_player.id)

    fetched_player = session.query(Player).filter_by(name = "Jonathan").first()
    print("Fetched from database:", fetched_player)

    session.close()