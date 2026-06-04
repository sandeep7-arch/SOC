# engine/board/move_history.py

class MoveHistory:
    """
    Stores all moves played in the game.
    """

    def __init__(self):
        self.moves = []

    def add_move(self, move):
        """
        Add move to history.
        """

        self.moves.append(move)

    def remove_last_move(self):
        """
        Remove most recent move.
        """

        if len(self.moves) > 0:
            return self.moves.pop()

        return None

    def last_move(self):
        """
        Get last move.
        """

        if len(self.moves) > 0:
            return self.moves[-1]

        return None

    def clear(self):
        """
        Clear history.
        """

        self.moves.clear()

    def total_moves(self):
        """
        Number of moves stored.
        """

        return len(self.moves)



history = MoveHistory()

history.add_move("e2e4")
history.add_move("e7e5")

print(history.last_move())
print(history.total_moves())
