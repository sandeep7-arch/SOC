from dataclasses import dataclass, field

@dataclass
class PlayerProfile:
    name: str
    rating: int = 1200  # default rating for a new player

    total_games: int = 0
    wins: int = 0
    losses : int = 0
    draws : int = 0

    # Weakness tracking — filled by vulnerability_vector.py
    weakness_scores: dict = field(default_factory= dict)

    # Style — filled by style_classifier.py
    playing_style = "Unknown"

    # Game ids — filled by game_index.py
    game_ids: list[str] = field(default_factory=list)

    def win_rate(self):
        if self.total_games == 0:
            return 0.0
        return round(self.wins / self.total_games, 2)
    
    def update_result(self, result):
        self.total_games += 1
        if result == "win":
            self.wins += 1
        elif result == "loss":
            self.losses += 1
        elif result == "draw":
            self.draws += 1

#demo test below
if __name__ == "__main__":
    player = PlayerProfile(name="Steve", rating=1200)
    print(player)
    print()

    player.update_result("win")
    player.update_result("win")
    player.update_result("loss")
    player.update_result("draw")

    print(f"Name: {player.name}")
    print(f"Total games: {player.total_games}")
    print(f"Wins: {player.wins}, Losses: {player.losses}, Draws: {player.draws}")
    print(f"Win rate: {player.win_rate()}")