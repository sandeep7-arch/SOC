import chess

def detect_forks(board, color):
    enemy_color = not color
    forks = []

    piece_values = {
        chess.KING : 100,
        chess.QUEEN : 9,
        chess.ROOK : 5,
        chess.BISHOP : 3,
        chess.KNIGHT : 3,
        chess.PAWN : 1
    }

    for square in chess.SQUARES:
        piece = board.piece_at(square)

        if not piece or piece.color != color:
            continue
        if piece.piece_type == chess.PAWN:
            continue

        attacked_squares = board.attacks(square)

        valuable_targets = []
        for attacked_sq in attacked_squares:
            target = board.piece_at(attacked_sq)
            if target and target.color == enemy_color:
                if piece_values.get(target.piece_type, 0) >= 3:
                    valuable_targets.append(attacked_sq)

        if len(valuable_targets) >= 2:
            forks.append({
                "piece": chess.piece_name(piece.piece_type),
                "square": chess.square_name(square),
                "attacking": [chess.square_name(sq) for sq in valuable_targets]
            })

    return forks

def detect_pins(board, color):
    enemy_color = not color
    pins = []

    for square in chess.SQUARES:
        piece = board.piece_at(square)

        if not piece or piece.color != color:
            continue

        if board.is_pinned(color, square):
            pins.append({
                "pinned_piece": chess.piece_name(piece.piece_type),
                "square": chess.square_name(square)
            })

    return pins

def detect_skewers(board, color):
    enemy_color = not color
    skewers = []

    piece_values = {
        chess.QUEEN:  9,
        chess.ROOK:   5,
        chess.BISHOP: 3,
        chess.KNIGHT: 3,
        chess.PAWN:   1,
        chess.KING:   100
    }

    sliding_pieces = [chess.BISHOP, chess.ROOK, chess.QUEEN]

    for square in chess.SQUARES:
        piece = board.piece_at(square)

        if not piece or piece.color != color:
            continue
        if piece.piece_type not in sliding_pieces:
            continue

        attacked_squares = board.attacks(square)

        for attacked_sq in attacked_squares:
            front_piece = board.piece_at(attacked_sq)

            if not front_piece or front_piece.color != enemy_color:
                continue

            if piece_values.get(front_piece.piece_type, 0) < 5:
                continue

            ray = chess.SquareSet(
                chess.BB_RAYS[square][attacked_sq]
            )

            found_back_piece = False
            for ray_sq in ray:
                if ray_sq == square or ray_sq == attacked_sq:
                    continue

                back_piece = board.piece_at(ray_sq)
                if back_piece:
                    if back_piece.color == enemy_color and \
                       piece_values.get(back_piece.piece_type, 0) < \
                       piece_values.get(front_piece.piece_type, 0):
                        skewers.append({
                            "attacker": chess.piece_name(piece.piece_type),
                            "attacker_square": chess.square_name(square),
                            "front_piece": chess.piece_name(front_piece.piece_type),
                            "front_square": chess.square_name(attacked_sq),
                            "back_piece": chess.piece_name(back_piece.piece_type),
                            "back_square": chess.square_name(ray_sq)
                        })
                        found_back_piece = True
                    break  

            if found_back_piece:
                break

    return skewers

def analyze_tactical_patterns(board, color):
    forks   = detect_forks(board, color)
    pins    = detect_pins(board, color)
    skewers = detect_skewers(board, color)

    return {
        "color": "white" if color == chess.WHITE else "black",
        "forks": forks,
        "pins": pins,
        "skewers": skewers,
        "total_tactics": len(forks) + len(pins) + len(skewers)
    }


if __name__ == "__main__":
    board = chess.Board("r3k3/2N5/8/8/8/8/8/4K3 w - - 0 1")
    print(board)
    print()

    result = analyze_tactical_patterns(board, chess.WHITE)
    print("White tactics:", result)
    print()

    result = analyze_tactical_patterns(board, chess.BLACK)
    print("Black tactics:", result)