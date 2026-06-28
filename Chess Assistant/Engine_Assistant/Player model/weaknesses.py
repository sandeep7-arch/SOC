from game_index import GameIndex

def count_weaknesses(game_index):
    # Counts how many times each weakness type appears across all games and returns a dict of {weakness_type: count}

    all_weaknesses = game_index.get_all_weaknesses()
    counts = {}

    for weakness in all_weaknesses:
        if weakness in counts:
            counts[weakness] += 1
        else:
            counts[weakness] = 1

    return counts

def get_most_common_weakness(counts):
    
    # Returns the weakness type that appears most frequently.
    
    if not counts:
        return None
    return max(counts, key=counts.get)


if __name__ == "__main__":
    from game_index import GameIndex, GameRecord

    index = GameIndex()
    index.add_game(GameRecord(
        game_id="game_001", result="loss", opponent_rating=1400,
        phase_reached="middlegame", total_moves=35,
        blunder_count=2, mistake_count=3, inaccuracy_count=4,
        weaknesses_found=["missed_tactic", "king_safety", "missed_tactic"]
    ))
    index.add_game(GameRecord(
        game_id="game_002", result="win", opponent_rating=1150,
        phase_reached="endgame", total_moves=52,
        blunder_count=1, mistake_count=1, inaccuracy_count=2,
        weaknesses_found=["endgame_error", "missed_tactic"]
    ))
    index.add_game(GameRecord(
        game_id="game_003", result="loss", opponent_rating=1300,
        phase_reached="middlegame", total_moves=28,
        blunder_count=3, mistake_count=2, inaccuracy_count=1,
        weaknesses_found=["king_safety", "positional_error", "missed_tactic"]
    ))

    counts = count_weaknesses(index)
    print("Weakness counts:", counts)
    print("Most common weakness:", get_most_common_weakness(counts))