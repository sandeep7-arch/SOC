def update_progress(history, current_vector):
    # History is a list of past vectors
    history.append(current_vector)
    return history

def get_trend(history, weakness_type):
    # Compares average of recent 3 games VS average of older games to check if the weakness is improving, worsening, or stable
    if len(history) < 2:
        return "not enough data"
    
    scores = [snapshot.get(weakness_type, 0.0) for snapshot in history]

    mid = len(scores)//2
    older_avg = sum(scores[:mid])/len(scores[:mid])
    recent_avg = sum(scores[mid:])/len(scores[mid:])

    diff = recent_avg - older_avg

    if diff < -0.1:
        return "improving"
    elif diff > 0.1:
        return "worsening"
    else:
        return "stable" 
    
def get_full_progress_report(history):
    # Returns trends for every weakness type.

    if not history:
        return {}
    
    all_types = list(history[0].keys())
    report = {}

    for weakness_type in all_types:
        report[weakness_type] = get_trend(history, weakness_type)

    return report
    
if __name__ == "__main__":

    history = []

    history = update_progress(history, {
        "missed_tactic": 1.0, "king_safety": 0.2,
        "positional_error": 0.5, "endgame_error": 0.3
    })
    history = update_progress(history, {
        "missed_tactic": 0.8, "king_safety": 0.4,
        "positional_error": 0.4, "endgame_error": 0.3
    })
    history = update_progress(history, {
        "missed_tactic": 0.6, "king_safety": 0.6,
        "positional_error": 0.4, "endgame_error": 0.2
    })
    history = update_progress(history, {
        "missed_tactic": 0.4, "king_safety": 0.8,
        "positional_error": 0.3, "endgame_error": 0.2
    })

    report = get_full_progress_report(history)
    print("Progress report:")
    for weakness, trend in report.items():
        print(f" {weakness:<25} {trend}")