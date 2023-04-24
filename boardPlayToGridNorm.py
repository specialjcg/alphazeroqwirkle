import numpy as np


def boardPlayToGridNorm(board, actions, playerval):
    nextBoard = np.zeros_like(board)
    nextBoard[:, :, :] = board[:, :, :]
    actions_array = np.array(actions)
    nextBoard[actions_array[:, 0] - 1, actions_array[:, 2] + 22, actions_array[:, 3] + 22] = playerval
    nextBoard[5 + actions_array[:, 1], actions_array[:, 2] + 22, actions_array[:, 3] + 22] = playerval
    return nextBoard
