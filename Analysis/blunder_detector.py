import chess

def blunder_detector(eval_before, eval_after, turn):
    if turn == chess.WHITE:
        drop = eval_before - eval_after
    else:
        drop = eval_after - eval_before
    
    if drop < 0.3:
        classification = "good"
    elif drop < 1.0:
        classification = "inaccuracy"
    elif drop < 1.5:
        classification = "mistake"
    else:
        classification = "blunder"
    
    return{"evaluation_before" : eval_before,
           "evaluation_after" : eval_after,
           "drop" : round(drop, 2),
           "classification" : classification
           }

if __name__ == "__main__":
    result = blunder_detector(
        eval_before=0.8,
        eval_after=-1.3,
        turn=chess.WHITE
    )
    print(result)

    result = blunder_detector(
        eval_before=0.3,
        eval_after=1.1,
        turn=chess.BLACK
    )
    print(result)

    result = blunder_detector(
        eval_before=0.5,
        eval_after=0.6,
        turn=chess.WHITE
    )
    print(result)
