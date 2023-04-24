import datetime
import pickle

import torch

from cnn_iter import ConnectNet
from convertToBoard import convertToBoard
from get_reward_for_player import get_reward_for_player
from get_canonical_board import get_canonical_board
from get_next_state import get_next_state
from Node import Node
from get_validMoves import get_valid_moves

cnn_iter1 = ConnectNet()
cnn_iter1.init_weights()

class MCTS_iter:

    def backpropagate(self, search_path, value, to_play):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        for node in reversed(search_path):
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1

    def run(self, state, to_play, action, game):
        with torch.no_grad():
            #   gridnormeOne = np.zeros(shape=(26,54,54))
            #   gridnormenegOne = np.zeros(shape=(6, 7))
            #   gridnormeZero = np.zeros(shape=(6, 7))
            root = Node(0, to_play, action)
            #   convert_zero(state, gridnormeZero)
            #   convert_one(state, gridnormeOne)
            #   convert_neg_one(state, gridnormenegOne)

            # statesignal = torch.tensor([gridAll], dtype=torch.float32).cuda()
            statesignal = torch.tensor(state, dtype=torch.float32)
            action_probs, value = cnn_iter1(statesignal)
            gamesimul = game.__copy__()
            gamesimul.actionprob = pickle.load(open('gameActionProb.pkl', 'rb'))
            valid_moves = get_valid_moves(gamesimul)

            # action_probs = action_probs * torch.tensor([valid_moves], dtype=torch.float32).cuda()
            action_probs = action_probs * valid_moves  # mask invalid moves
            # action_probs = add_dirichlet_noise(action_probs)

            action_probs /= torch.sum(action_probs)

            root.expand(state, to_play, torch.squeeze(action_probs, 0))
            if len(root.children) > 0:

                start_time = datetime.datetime.now().time().strftime('%H:%M:%S')

                # for i in range(777):
                for i in range(600):

                    node = root.copy()
                    search_path = [node]

                    # SELECT
                    while node.expanded():
                        action, node = node.select_child(gamesimul)
                        search_path.append(node)

                    parent = search_path[-2]
                    state = parent.state
                    # Now we're at a leaf node and we would like to expand
                    # Players always play from their own perspective

                    next_state, _, isLegalmove = get_next_state(state, player=to_play, action=action, game=gamesimul)
                    # Get the board from the perspective of the other player
                    next_state = get_canonical_board(next_state, player=-to_play)

                    # The value of the new state from the perspective of the other player
                    value = get_reward_for_player(next_state, player=to_play)
                    if value == 0 and len(isLegalmove) != 0:
                        # If the game has not ended:
                        # EXPAND
                        if (parent.to_play == 1):
                            if len(gamesimul.listValidMoves) > 0:
                                for tile in isLegalmove:
                                    gamesimul.place(tile[0], tile[1], tile[2], tile[3])
                                    gamesimul.player1.delRack(tile[0], tile[1])
                                gamesimul.player1.addTileToRack(gamesimul.bag)
                                gamesimul.listValidMovePlayer1()
                                next_state = convertToBoard(next_state, gamesimul.player1.getRack())

                            else:
                                gamesimul.player1.newRack(gamesimul.bag)



                        else:

                            if len(gamesimul.listValidMoves) > 0:
                                for tile in isLegalmove:
                                    gamesimul.place(tile[0], tile[1], tile[2], tile[3])
                                    gamesimul.player2.delRack(tile[0], tile[1])
                                gamesimul.player2.addTileToRack(gamesimul.bag)
                                gamesimul.listValidMovePlayer2()
                                next_state = convertToBoard(next_state, gamesimul.player2.getRack())
                            else:
                                gamesimul.player2.newRack(gamesimul.bag)

                        gridAll = next_state
                        # statesignal = torch.tensor([gridAll], dtype=torch.float32).cuda()
                        statesignal = torch.tensor(gridAll, dtype=torch.float32)
                        action_probs, value = cnn_iter1(statesignal)

                        valid_moves = get_valid_moves(gamesimul)

                        # action_probs = action_probs * torch.tensor([valid_moves], dtype=torch.float32).cuda()
                        action_probs = action_probs * valid_moves  # mask invalid moves
                        # action_probs = add_dirichlet_noise(action_probs)

                        action_probs /= torch.sum(action_probs)

                        node.expand(next_state, parent.to_play * -1, torch.squeeze(action_probs, 0))

                    self.backpropagate(search_path, value, parent.to_play * -1)
                end_time = datetime.datetime.now().time().strftime('%H:%M:%S')
                total_time = (datetime.datetime.strptime(end_time, '%H:%M:%S') - datetime.datetime.strptime(start_time,
                                                                                                            '%H:%M:%S'))
                print('total_time:' + str(total_time))
                return root
            else:
                return root
