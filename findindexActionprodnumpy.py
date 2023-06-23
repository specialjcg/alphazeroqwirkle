# def findindexinActionprobnumpy(game):
#     valid_moves = np.zeros(23436)
#     for i in game.listValidMoves:
#         valprob = [[testnumpy[0], testnumpy[1]] for testnumpy in i]
#         for index, x in enumerate(game.actionprob):
#             if x == tuple(valprob):
#                 valid_moves[index] = 1
#     return valid_moves
import pandas as pd
import numpy as np

from decompose_list import decompose_list


def findindexinActionprobnumpy(game):
    df = pd.DataFrame(game.listValidMoves)
    df = df.applymap(lambda x: x[:2] if isinstance(x, list) else x).fillna(0)
    df = df.apply(lambda x: pd.Series(decompose_list(x),dtype='int64'), axis=1).fillna(0)
    new_df = pd.DataFrame(0, index=df.index, columns=range(len(df.columns), 12))
    # concatenate the two dataframes along the columns axis
    df = pd.concat([df, new_df], axis=1)
    valid_moves = np.zeros(23436)
    # Convert dataframes to numpy arrays for faster comparison
    df_np = df.to_numpy()
    df1_np = game.df1.to_numpy()
    # Create boolean mask of matching rows
    matching_rows = np.all(df_np[:, np.newaxis, :] == df1_np, axis=-1)
    # Set corresponding indices of valid_moves to 1
    valid_moves[matching_rows.any(axis=0)] = 1
    return valid_moves

