from boardPlayToGridNorm import boardPlayToGridNorm


def get_next_state(board, player, action, game):
    nextState = list(game.actionprob[action])
    result=[tile for tile in game.listValidMoves if len(tile) == 1 and tile[0][0:2] == nextState[0] and all(abs(tilecoord[2]) < 22 and abs(tilecoord[3]) < 22 for tilecoord in tile)]
     # for tiles in game.listValidMoves:
     #    if all(abs(tile[2]) < 22 and abs(tile[3]) < 22 for tile in tiles) and nextState == [[tile[0], tile[1]] for tile
     #                                                                                        in tiles]:
     #        return boardPlayToGridNorm(board, tiles, 1), -player, tiles
    if result!=[]:
        return boardPlayToGridNorm(board, result[0], 1), -player, result[0]
    return board, -player, []


