import chess

def get_piece_activity(board, color):
    total_mobility = 0
    piece_count = 0

    for square in chess.SQUARES:
        piece = board.piece_at(square)

        if not piece or piece.color != color:
            continue
        if piece.piece_type == chess.KING:
            continue
        if piece.piece_type == chess.PAWN:
            continue

        attacked = len(board.attacks(square))
        total_mobility += attacked
        piece_count +=1

    if piece_count == 0:
        return 0.0
    
    return round(total_mobility/piece_count, 2)

def get_space_control(board, color):
    controlled_squares = 0

    for square in chess.SQUARES:
        if board.is_attacked_by(color, square):
            controlled_squares += 1

    return controlled_squares


def get_rooks_on_open_files(board, color):
    open_rooks = []

    for square in board.pieces(chess.ROOK, color):
        file = chess.square_file(square)
        is_open = True

        for rank in range(8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            if piece and piece.piece_type == chess.PAWN:
                is_open = False
                break

        if is_open:
            open_rooks.append(chess.square_name(square))

    return open_rooks


def get_center_control(board, color):
    center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
    controlled = 0

    for square in center_squares:
        if board.is_attacked_by(color, square):
            controlled += 1

    return controlled


def analyze_positional(board, color):
    activity      = get_piece_activity(board, color)
    space         = get_space_control(board, color)
    open_rooks    = get_rooks_on_open_files(board, color)
    center        = get_center_control(board, color)

    positional_score = round((activity / 10) + (center * 0.5) + (len(open_rooks) * 0.5), 2)

    return {
        "color": "white" if color == chess.WHITE else "black",
        "piece_activity": activity,
        "space_control": space,
        "center_control": center,
        "rooks_on_open_files": open_rooks,
        "positional_score": positional_score
    }


if __name__ == "__main__":
    board = chess.Board()
    print("Starting position:")
    print(board)
    print()

    result = analyze_positional(board, chess.WHITE)
    print("White positional:", result)
    print()

    result = analyze_positional(board, chess.BLACK)
    print("Black positional:", result)
    print()

    board.push_san("e4")
    print("After 1.e4:")
    result = analyze_positional(board, chess.WHITE)
    print("White positional:", result)
    print()
    result = analyze_positional(board, chess.BLACK)
    print("Black positional:", result)