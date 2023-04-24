import numpy as np


def isFinish(gridnorme, game):
    matrix = np.array(gridnorme)
    # Count occurrence of element '3' in each column
    count = np.count_nonzero(matrix == 1)
    return game.bag.isEmpty() and (len(game.player1.rack) == 0 or len(game.player2.rack) == 0)
