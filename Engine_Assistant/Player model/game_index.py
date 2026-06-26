from dataclasses import dataclass, field

@dataclass
class GameRecord:
    game_id: str
    result : str
    opponent_rating : str
    phase_reached : str
    total_moves : int
    blunder_count : int
    mistake_count : int
    inaccuracy_count : int
    weaknesses_found : list = field(default_factory=list)


class GameIndex:
    def __init__(self):
        self.games: list[GameRecord] = []

    def add_game(self, game_record):
        self.games.append(game_record)

    def total_games(self):
        return len(self.games)
    
    def get_all_weaknesses(self):

        #Returns flat list of all weaknesses across all games.
        all_weaknesses = []
        for game in self.games:
            all_weaknesses.extend(game.weaknesses_found)
        return all_weaknesses
    
    def get_recent_games(self, n=5):
        """
        Returns the last n games played.
        More recent games matter more for the player model.
        """
        return self.games[-n:]
    
    def average_blunders_per_game(self):
        if not self.games:
            return 0.0
        total = sum(g.blunder_count for g in self.games)
        return round(total/len(self.games), 2)
    
if __name__ == "__main__":
    index = GameIndex()

    # Simulate adding a few games
    index.add_game(GameRecord(
        game_id="game_001",
        result="loss",
        opponent_rating=1400,
        phase_reached="middlegame",
        total_moves=35,
        blunder_count=2,
        mistake_count=3,
        inaccuracy_count=4,
        weaknesses_found=["missed_tactic", "king_safety", "missed_tactic"]
    ))

    index.add_game(GameRecord(
        game_id="game_002",
        result="win",
        opponent_rating=1150,
        phase_reached="endgame",
        total_moves=52,
        blunder_count=1,
        mistake_count=1,
        inaccuracy_count=2,
        weaknesses_found=["endgame_error", "missed_tactic"]
    ))

    index.add_game(GameRecord(
        game_id="game_003",
        result="loss",
        opponent_rating=1300,
        phase_reached="middlegame",
        total_moves=28,
        blunder_count=3,
        mistake_count=2,
        inaccuracy_count=1,
        weaknesses_found=["king_safety", "positional_error", "missed_tactic"]
    ))

    print(f"Total games: {index.total_games()}")
    print(f"Avg blunders per game: {index.average_blunders_per_game()}")
    print(f"All weaknesses: {index.get_all_weaknesses()}")
    print(f"Recent 2 games: {index.get_recent_games(2)}")