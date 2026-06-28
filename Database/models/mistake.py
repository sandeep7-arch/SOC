import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship
from base import Base


class Mistake(Base):
    # Stores individual mistakes detected during game analysis. Each row denotes in mistake in one move of one game

    __tablename__ = "mistakes"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Links to which game this mistake happened in
    game_id = Column(Integer, ForeignKey("games.id"), nullable=False)
    
    # Links to which player made the mistake
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)

    move_number = Column(Integer)
    move = Column(String)
    classification = Column(String)
    category = Column(String)
    eval_before = Column(Float)
    eval_after = Column(Float)
    drop = Column(Float)
    phase = Column(String)

    game = relationship("Game", back_populates="mistakes")
    player = relationship("Player", back_populates="mistakes")

    def __repr__(self):
        return(f"<Mistake(move={self.move}, "
               f"classification = {self.classification},"
               f"category={self.category},"
               f"drop={self.drop})>")
    
if __name__ == "__main__":
    from base import init_db, get_session
    
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
        blunder_count = 1,
        mistake_count = 1,
        inaccuracy_count = 2
    )
    session.add(game)
    session.commit()

    mistake = Mistake(
        game_id = game.id,
        player_id = player.id,
        move_number = 23,
        move = "Nd4",
        classification = "blunder",
        category = "missed_tactic",
        eval_before = 0.8,
        eval_after = -1.3,
        drop = 2.1,
        phase = "middlegame"
    )
    session.add(mistake)
    session.commit()

    print("Mistake saved:", mistake)

    blunders = session.query(Mistake).filter_by(
        player_id = player.id,
        classification = "blunder"
    ).all()
    print(f"All blunders for {player.name}: {blunders}")

    missed_tactics = session.query(Mistake).filter_by(
        player_id = player.id,
        category = "missed_tactic"
    ).all()
    print(f"Missed tactics for {player.name}: {missed_tactics}")

    session.commit()
    session.close()