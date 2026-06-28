import chess

def get_game_phase(board):
    
    queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))
    rooks = len(board.pieces(chess.ROOK, chess.WHITE)) + len(board.pieces(chess.ROOK, chess.BLACK))
    knights = len(board.pieces(chess.KNIGHT, chess.WHITE)) + len(board.pieces(chess.KNIGHT, chess.BLACK))
    bishops = len(board.pieces(chess.BISHOP, chess.WHITE)) + len(board.pieces(chess.BISHOP, chess.BLACK))

    material_score = (queens*9) + (rooks*5) + (knights*3) + (bishops*3)

    move_count = len(board.move_stack)

    if move_count < 10:
        return "opening"
    elif material_score <= 14:
        return "endgame"
    else:
        return "middlegame"


if __name__ == "__main__":
    board = chess.Board()
    print(get_game_phase(board))

    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Nf3")
    board.push_san("Nc6")
    board.push_san("Bc4")
    board.push_san("Bc5")
    board.push_san("O-O")
    board.push_san("Nf6")
    board.push_san("d3")
    board.push_san("d6")
    board.push_san("Nc3")

    print(get_game_phase(board))