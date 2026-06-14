def update_vulnerability_vector(old_vector, new_game_vector, weight = 0.3):
    updated_vector = {}

    for weakness_type in old_vector:
        old_score = old_vector[weakness_type]
        new_score = new_game_vector.get(weakness_type, 0.0)

        updated_score = (old_score * (1 - weight)) + (new_score * weight)
        updated_vector[weakness_type] = round(updated_score, 2)

    return updated_vector

if __name__ == "__main__":
    old_vector = {
        "missed_tactic": 0.8,
        "king_safety": 0.3,
        "positional_error": 0.5,
        "endgame_error": 0.2
    }

    new_game_vector = {
        "missed_tactic": 0.0,
        "king_safety": 1.0,
        "positional_error": 0.5,
        "endgame_error": 0.0
    }

    updated = update_vulnerability_vector(old_vector, new_game_vector, weight=0.3)

    print("Old vector:")
    for k, v in old_vector.items():
        print(f"  {k:<20} {v}")

    print("\nNew game vector:")
    for k, v in new_game_vector.items():
        print(f"  {k:<20} {v}")

    print("\nUpdated vector (weight=0.3):")
    for k, v in updated.items():
        print(f"  {k:<20} {v}")