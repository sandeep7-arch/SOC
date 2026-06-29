def classify_style(vulnerability_vector, game_index):
    if game_index.total_games() == 0:
        return {"Style" : "Unknown","Confidence" : 0.0}
    
    missed_tactic = vulnerability_vector.get("missed_tactic", 0.0)
    king_safety = vulnerability_vector.get("king_safety", 0.0)
    positional_error = vulnerability_vector.get("positional_error", 0.0)
    endgame_error = vulnerability_vector.get("endgame_error", 0.0)

    avg_blunders = game_index.average_blunders_per_game()

    scores = {
        "aggressive" : 0.0,
        "defensive" : 0.0,
        "tactical" : 0.0,
        "positional" : 0.0
    }

    # Aggressive players tend to have higher blunder rates and higher king safety issues
    if avg_blunders > 1.5:
        scores["aggressive"] +=0.5
    if king_safety>0.5:
        scores["aggressive"] +=0.3

    # Defensive players have low blunder rates and good king safety
    if avg_blunders < 1.0:
        scores["defensive"] += 0.5
    if king_safety < 0.3:
        scores["defensive"] += 0.3

     # Tactical players have low missed tactic scores
    if missed_tactic < 0.3:
        scores["tactical"] += 0.6
    if avg_blunders < 1.2:
        scores["tactical"] += 0.2

    # Positional players have low positional errors
    if positional_error < 0.3:
        scores["positional"] += 0.5
    if endgame_error < 0.3:
        scores["positional"] += 0.3

    # Pick the style with highest score
    best_style = max(scores, key=scores.get)
    confidence = scores[best_style]

    return {
        "style": best_style,
        "confidence": round(confidence, 2),
        "all_scores": scores
    }

if __name__ == "__main__":
    from game_index import GameIndex, GameRecord

    index = GameIndex()
    index.add_game(GameRecord(
        game_id="game_001", result="win", opponent_rating=1300,
        phase_reached="middlegame", total_moves=30,
        blunder_count=2, mistake_count=1, inaccuracy_count=1,
        weaknesses_found=["king_safety", "missed_tactic"]
    ))
    index.add_game(GameRecord(
        game_id="game_002", result="loss", opponent_rating=1300,
        phase_reached="middlegame", total_moves=25,
        blunder_count=3, mistake_count=2, inaccuracy_count=1,
        weaknesses_found=["king_safety", "king_safety"]
    ))

    vulnerability_vector = {
        "missed_tactic": 0.5,
        "king_safety": 1.0,
        "positional_error": 0.2,
        "endgame_error": 0.1,
        "opening_error": 0.0,
        "doubled_pawns": 0.0,
        "isolated_pawns": 0.0,
        "low_piece_activity": 0.0,
        "poor_center_control": 0.0
    }

    result = classify_style(vulnerability_vector, index)
    print("Style classification:")
    print(f"  Style: {result['style']}")
    print(f"  Confidence: {result['confidence']}")
    print(f"  All scores: {result['all_scores']}")