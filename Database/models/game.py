import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship
from base import Base
from player import Player

class Game(Base):
    # Database table for storing game records. Similar to game_index.py but stores permanently

    __tablename__ = "games"
    id = Column(Integer, primary_key=True, autoincrement= True)

    # Foreign key links this game to a player in the players table
    player_id = Column(Integer, ForeignKey("players.id"), nullable= False)

    result = Column(String, nullable=False)
    opponent_rating = Column(Integer)
    phase_reached = Column(Integer)
    total_moves = Column(Integer)
    blunder_count = Column(Integer, default = 0)
    mistake_count = Column(Integer, default = 0)
    inaccuracy_count = Column(Integer, default = 0)
    
    # Relationship lets access player directly from game object
    player = relationship("Player", back_populates = "games")

    def __repr__(self):
        return f"<Game(player_id = {self.player_id}, result = {self.result}, blunders = {self.blunder_count})>"
    
    mistakes = relationship("Mistake", back_populates="game")
    
if __name__ == "__main__":
    from base import init_db, get_session
    from player import Player

    init_db()
    session = get_session()

    player = Player(name="Jonathan", rating = 1200)
    session.add(player)
    session.commit()

    game = Game(
        player_id = player.id,
        result = "loss",
        opponent_rating = 1400,
        phase_reached = "middlegame",
        total_moves = 35,
        blunder_count = 2,
        mistake_count = 3,
        inaccuracy_count = 4
    )
    session.add(game)
    session.commit()

    print("Game saved:", game)
    print("Game Id:", game.id)
    print("linked to player ID:", game.player_id)


    player_games = session.query(Game).filter_by(player_id = player.id).all()
    print(f"Games for player {player.name}: {player_games}")

    session.commit()
    session.close()
