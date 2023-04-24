import numpy as np


def convertToBoard(state, racks):
    nextBoard = np.zeros_like(state)
    nextBoard[:12, :] = state[:12, :]
    for rack in racks:
        nextBoard[12 + rack[0], 0:len(rack)] = 1
        nextBoard[18 + rack[1], 0:len(rack)] = 1
    return nextBoard
