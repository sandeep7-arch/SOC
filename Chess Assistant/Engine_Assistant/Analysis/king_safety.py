import chess

def get_king_zone(board, color):
    king_square = board.king(color)
    
    if king_square is None:
        return set()
    
    king_zone = set()
    king_zone.add(king_square)
    
    for sq in chess.SquareSet(chess.BB_KING_ATTACKS[king_square]):
        king_zone.add(sq)
    
    return king_zone


def get_pawn_shield_score(board, color):
    king_square = board.king(color)
    
    if king_square is None:
        return 0
    
    king_file = chess.square_file(king_square)
    king_rank = chess.square_rank(king_square)
    
    shield_count = 0
    
    for file_offset in [-1, 0, 1]:
        adjacent_file = king_file + file_offset
        
        if not (0 <= adjacent_file <= 7):
            continue
        
        if color == chess.WHITE:
            shield_rank = king_rank + 1
        else:
            shield_rank = king_rank - 1
        
        if not (0 <= shield_rank <= 7):
            continue
        
        shield_square = chess.square(adjacent_file, shield_rank)
        piece = board.piece_at(shield_square)
        
        if piece and piece.piece_type == chess.PAWN and piece.color == color:
            shield_count += 1
    
    return shield_count


def get_open_files_near_king(board, color):
    king_square = board.king(color)
    
    if king_square is None:
        return []
    
    king_file = chess.square_file(king_square)
    open_files = []
    
    for file_offset in [-1, 0, 1]:
        check_file = king_file + file_offset
        
        if not (0 <= check_file <= 7):
            continue
        
        has_pawn = False
        for rank in range(8):
            square = chess.square(check_file, rank)
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN:
                has_pawn = True
                break
        
        if not has_pawn:
            open_files.append(chess.FILE_NAMES[check_file])
    
    return open_files


def get_attackers_on_king_zone(board, color):
    enemy_color = not color
    king_zone = get_king_zone(board, color)
    
    attackers = set()
    
    for square in king_zone:
        attacking_squares = board.attackers(enemy_color, square)
        for attacker_sq in attacking_squares:
            attackers.add(attacker_sq)
    
    return len(attackers)


def analyze_king_safety(board, color):
    pawn_shield = get_pawn_shield_score(board, color)
    open_files  = get_open_files_near_king(board, color)
    attackers   = get_attackers_on_king_zone(board, color)
    
    safety_score = pawn_shield - len(open_files) - (attackers * 0.5)
    
    # Classify safety level
    if safety_score >= 2:
        safety_level = "safe"
    elif safety_score >= 0:
        safety_level = "slightly exposed"
    else:
        safety_level = "dangerous"
    
    return {
        "color": "white" if color == chess.WHITE else "black",
        "pawn_shield": pawn_shield,
        "open_files_near_king": open_files,
        "attacker_count": attackers,
        "safety_score": round(safety_score, 2),
        "safety_level": safety_level
    }


if __name__ == "__main__":
    board = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 0 1")
    print(board)
    print()
    
    result = analyze_king_safety(board, chess.WHITE)
    print("White king safety:", result)
    print()
    
    result = analyze_king_safety(board, chess.BLACK)
    print("Black king safety:", result)