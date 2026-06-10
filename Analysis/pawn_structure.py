import chess


def get_doubled_pawns(board, colour):
    doubled = []

    for file in range(8):
        pawns_on_file = 0

        for rank in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)

            if piece and piece.piece_type == chess.PAWN and piece.color == colour:  
                pawns_on_file += 1

        if pawns_on_file > 1:
            doubled.append(chess.FILE_NAMES[file])  

    return doubled


def get_isolated_pawns(board, colour):
    isolated = []

    files_with_pawns = set()
    for square in board.pieces(chess.PAWN, colour):
        files_with_pawns.add(chess.square_file(square))

    for file in files_with_pawns:
        left_neighbour = file - 1
        right_neighbour = file + 1

        has_neighbour = (left_neighbour in files_with_pawns) or (right_neighbour in files_with_pawns)

        if not has_neighbour:
            isolated.append(chess.FILE_NAMES[file])

    return isolated  


def get_passed_pawns(board, colour):
    passed = []
    enemy_colour = not colour

    for square in board.pieces(chess.PAWN, colour):
        file = chess.square_file(square)
        rank = chess.square_rank(square)

        is_passed = True

        ahead_range = range(rank + 1, 8) if colour == chess.WHITE else range(rank - 1, -1, -1)  

        for ahead_rank in ahead_range:
            for adjacent_file in [file - 1, file, file + 1]:
                if 0 <= adjacent_file <= 7:
                    sq = chess.square(adjacent_file, ahead_rank)
                    piece = board.piece_at(sq)
                    if piece and piece.piece_type == chess.PAWN and piece.color == enemy_colour:  
                        is_passed = False
                        break

        if is_passed:
            passed.append(chess.square_name(square))

    return passed


def analyze_pawn_structure(board, colour):
    doubled = get_doubled_pawns(board, colour)
    isolated = get_isolated_pawns(board, colour)
    passed = get_passed_pawns(board, colour)

    return {
        "colour": "white" if colour == chess.WHITE else "black",
        "doubled_pawns": doubled,
        "isolated_pawns": isolated,
        "passed_pawns": passed
    }


if __name__ == "__main__":
    board = chess.Board("8/8/8/8/8/2P5/2P3p1/8 w - - 0 1")
    print(board)
    print()

    result = analyze_pawn_structure(board, chess.WHITE)
    print("White pawn structure:", result)

    result = analyze_pawn_structure(board, chess.BLACK)
    print("Black pawn structure:", result)
