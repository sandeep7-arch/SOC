from move_history import MoveHistory

history = MoveHistory()

history.add_move("e2e4")
history.add_move("e7e5")

print(history.last_move())
print(history.total_moves())