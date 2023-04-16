import concurrent.futures
import os
import pickle
from collections import namedtuple, deque

import numpy as np
import torch

import GameNumpy as newGame
from TileColor import TileColor
from TileShape import TileShape
from cnn import ConnectNet

cuda0 = torch.device('cuda:0')

global epoch
epoch = 0
import random
import math
import logging

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


# def boardPlayToGridNorm(board, actions, playerval):
#     nextBoard = np.copy(board)
#     actions_array = np.array(actions)
#     nextBoard[actions_array[:, 0]-1, actions_array[:, 2]+22, actions_array[:, 3]+22] = playerval
#     nextBoard[5+actions_array[:, 1], actions_array[:, 2]+22, actions_array[:, 3]+22] = playerval
#     return nextBoard
def boardPlayToGridNorm(board, actions, playerval):
    nextBoard = np.zeros_like(board)
    nextBoard[:, :, :] = board[:, :, :]
    actions_array = np.array(actions)
    nextBoard[actions_array[:, 0] - 1, actions_array[:, 2] + 22, actions_array[:, 3] + 22] = playerval
    nextBoard[5 + actions_array[:, 1], actions_array[:, 2] + 22, actions_array[:, 3] + 22] = playerval
    return nextBoard


# def boardPlayToGridNorm(board, actions, playerval):
#     nextBoard = np.copy(board)
#     for rack in actions:
#         print(rack)
#         nextBoard[rack[0]-1][rack[2]+22][rack[3]+22]=playerval
#         nextBoard[5+rack[1]][rack[2]+22][rack[3]+22]=playerval
#     return nextBoard

def gridNormtoRack(gridnorme):
    nextBoard = np.copy(gridnorme)
    board = []
    for x in range(-22, 22):
        for y in range(-22, 22):
            for j in range(13, 19):
                if nextBoard[j][x + 22, y + 22] == 1.0 or nextBoard[j][x + 22, y + 22] == -1.0:
                    for k in range(19, 26):
                        if nextBoard[k][x + 22, y + 22] == 1.0 or nextBoard[k][x + 22, y + 22] == -1.0:
                            board.append([list(TileColor.keys())[j - 13], list(TileShape.keys())[k - 19], [0, 0]])
    return board


def gridNormToBoardPlay(gridnorme):
    nextBoard = np.copy(gridnorme)
    board = []
    for x in range(-22, 22):
        for y in range(-22, 22):
            for j in range(0, 6):
                if nextBoard[j][x + 22, y + 22] == 1.0 or nextBoard[j][x + 22, y + 22] == -1.0:
                    for k in range(6, 13):
                        if nextBoard[k][x + 22, y + 22] == 1.0 or nextBoard[k][x + 22, y + 22] == -1.0:
                            board.append([list(TileColor.keys())[j], list(TileShape.keys())[k - 6], [x, y]])
    return board




# def get_next_state(board, player, action,game):
#     nextState=list(game.actionprob[action])
#
#     for tiles in game.listValidMoves:
#         if all(abs(x) < 22 and abs(y) < 22 for x, y in [(tile[2], tile[3]) for tile in tiles]):
#             if nextState == [[tile[0], tile[1]] for tile in tiles]:
#                 return boardPlayToGridNorm(board, tiles, 1), -player, tiles
#
#     return board, -player,[]


def get_next_state(board, player, action, game):
    nextState = list(game.actionprob[action])
    for tiles in game.listValidMoves:
        if all(abs(tile[2]) < 22 and abs(tile[3]) < 22 for tile in tiles) and nextState == [[tile[0], tile[1]] for tile
                                                                                            in tiles]:
            return boardPlayToGridNorm(board, tiles, 1), -player, tiles
    return board, -player, []


# def get_next_state(board, player, action, game):
#     nextState = list(game.actionprob[action])
#
#     for tiles in game.listValidMoves:
#         if all(abs(x) < 22 and abs(y) < 22 for x, y in [(tile[2], tile[3]) for tile in tiles]):
#             if nextState == [[tile[0], tile[1]] for tile in tiles]:
#                 return boardPlayToGridNorm(board, tiles, 1), -player, tiles
#
#     return board, -player, []

# for tiles in game.listValidMoves:
#     tile_positions = [(tile[2], tile[3]) for tile in tiles]
#     if any(abs(x) >= 22 or abs(y) >= 22 for x, y in tile_positions):
#         continue
#     if nextState == [(tile[0], tile[1]) for tile in tiles]:
#         return boardPlayToGridNorm(board, tiles, 1), -player, tiles

# Return the new game, but
# change the perspective of the game with negative


# def findindexinActionprob(game):
#     valid_moves = np.zeros(len(game.actionprob))
#     for i in game.listValidMoves:
#         j=0
#         for element in game.actionprob:
#             if len(i)==len(element):
#                 if (element[0][0] == i[0][0] or element[0][1] == i[0][1]):
#                     for action in element:
#
#                         if action[2]==i[0][2] and action[3]==i[0][3]:
#                             valid_moves[j] = 1
#                         else:
#                             valid_moves[j] = 0
#                             break
#
#                 else:
#                     valid_moves[j] = 0
#             j+=1
#     return valid_moves
def findindexinActionprob(game):
    valid_moves = np.zeros(len(game.actionprob))
    j = 0
    for i, element in zip(game.listValidMoves, game.actionprob):
        if len(i) == len(element):
            if element[0][0] == i[0][0] or element[0][1] == i[0][1]:
                for action in element:
                    valid_moves[j] = 1 if action[2] == i[0][2] and action[3] == i[0][3] else 0
        else:
            valid_moves[j] = 0
        j += 1
    return valid_moves


def findindexinActionprobnumpy(game):
    valid_moves = np.zeros(23436)
    for i in game.listValidMoves:
        valprob = [[testnumpy[0], testnumpy[1]] for testnumpy in i]
        for index, x in enumerate(game.actionprob):
            if x == tuple(valprob):
                valid_moves[index] = 1
    return valid_moves


def get_valid_moves(game):
    return findindexinActionprobnumpy(game)


def get_reward_for_player(board, player):
    return 0


def get_canonical_board(board, player):
    return player * board


# In[245]:


def add_dirichlet_noise(child_priors):
    dirichlet_input = [0.1 for x in range(len(child_priors[0]))]
    dirichlet_list = np.random.dirichlet(dirichlet_input)
    noisy_psa_vector = []
    for idx, psa in enumerate(child_priors):
        noisy_psa_vector.append(
            (1 - 0.25) * psa + 0.25 * dirichlet_list[idx])
    child_priors = noisy_psa_vector[0]

    return (child_priors)
import torch
from torch import nn as nn
from torch.nn import functional as F


class ConvBlockiter(nn.Module):
    def __init__(self):
        super(ConvBlockiter, self).__init__()

        self.conv1 = nn.Conv2d(26, 1024 , kernel_size=12, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.conv2 = nn.Conv2d(1024, 35, kernel_size=12, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(35)




    def forward(self, s):
        s = s.view(-1,26, 54, 54)  # batch_size x channels x board_x x board_y
        s = F.leaky_relu(self.bn1(self.conv1(s)))
        s = F.leaky_relu(self.bn2(self.conv2(s)))

        return s


class ResBlockiter(nn.Module):
    def __init__(self, inplanes=35  , planes=35, stride=1, downsample=None):
        super(ResBlockiter, self).__init__()
        #self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
        #                       padding=1, bias=False).cuda()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        #self.bn1 = nn.BatchNorm2d(planes).cuda()
        self.bn1 = nn.BatchNorm2d(planes)
        #self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
        #                       padding=1, bias=False).cuda()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        #self.bn2 = nn.BatchNorm2d(planes).cuda()
        #self.drp = nn.Dropout(0.3).cuda()
        self.bn2 = nn.BatchNorm2d(planes)
        self.drp = nn.Dropout(0.3)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.leaky_relu(self.bn1(out))
        out = self.drp(out)
        out = self.conv2(out)
        out = F.leaky_relu(self.bn2(out))
        out = self.drp(out)
        out += residual
        out = F.leaky_relu(out)
        return out


class OutBlockiter(nn.Module):
    # shape=6*7*32
    shape1 = 45360
    # shape = 24156*25*25
    shape = 23436

    def __init__(self):
        super(OutBlockiter, self).__init__()


        self.fc1 = nn.Linear(self.shape1,1024)
        self.fc2 = nn.Linear(1024, 1)
        self.drp = nn.Dropout(0.3)

        self.fc = nn.Linear(self.shape1, self.shape1)
        self.fcinter = nn.Linear(self.shape1, self.shape)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, s):
        v = s.view(-1, self.shape1)  # batch_size X channel X height X width
        v = self.drp(F.leaky_relu(self.fc1(v)))
        v = torch.tanh(self.fc2(v))




        p = s.view(-1, self.shape1)
        p = self.drp(F.leaky_relu(self.fc(p)))
        p = self.fcinter(p)
        p = self.logsoftmax(p).exp()
        return p, v


class ConnectNetiter(nn.Module):
    def __init__(self):
        super(ConnectNetiter, self).__init__()
        #self.conv = ConvBlock().cuda()
        self.conv = ConvBlockiter()
        for block in range(30):
            #setattr(self, "res_%i" % block, ResBlock().cuda())
            setattr(self, "res_%i" % block, ResBlockiter())
        self.outblock = OutBlockiter()

    def forward(self, s):
        s = self.conv(s)
        for block in range(30):
            s = getattr(self, "res_%i" % block)(s)
        s = self.outblock(s)
        return s

    def init_weights(self):
        """
        Initialize weights of layers using Kaiming Normal (He et al.) as argument of "Apply" function of
        "nn.Module"
        :param m: Layer to initialize
        :return: None
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # if isinstance(self.conv.conv1, nn.Conv2d):
        #     torch.nn.init.xavier_uniform_(self.conv.conv1.weight)
        #     torch.nn.init.zeros_(self.conv.conv1.bias)
        # if isinstance(self.conv.bn1, nn.BatchNorm2d):
        #     torch.nn.init.normal_(self.conv.bn1.weight.data, mean=1, std=0.02)
        #     torch.nn.init.constant_(self.conv.bn1.bias.data, 0)
        # if isinstance(self.conv.conv2, nn.Conv2d):
        #     torch.nn.init.xavier_uniform_(self.conv.conv2.weight)
        #     torch.nn.init.zeros_(self.conv.conv2.bias)
        # if isinstance(self.conv.bn2, nn.BatchNorm2d):
        #     torch.nn.init.normal_(self.conv.bn2.weight.data, mean=1, std=0.02)
        #     torch.nn.init.constant_(self.conv.bn2.bias.data, 0)
        # if isinstance(self.conv.conv3, nn.Conv2d):
        #     torch.nn.init.xavier_uniform_(self.conv.conv3.weight)
        #     torch.nn.init.zeros_(self.conv.conv3.bias)
        # if isinstance(self.conv.bn3, nn.BatchNorm2d):
        #     torch.nn.init.normal_(self.conv.bn3.weight.data, mean=1, std=0.02)
        #     torch.nn.init.constant_(self.conv.bn3.bias.data, 0)
        # if isinstance(self.conv.conv4, nn.Conv2d):
        #     torch.nn.init.xavier_uniform_(self.conv.conv4.weight)
        #     torch.nn.init.zeros_(self.conv.conv4.bias)
        # if isinstance(self.conv.bn4, nn.BatchNorm2d):
        #     torch.nn.init.normal_(self.conv.bn4.weight.data, mean=1, std=0.02)
        #     torch.nn.init.constant_(self.conv.bn4.bias.data, 0)
        # if isinstance(self.conv.conv5, nn.Conv2d):
        #     torch.nn.init.xavier_uniform_(self.conv.conv5.weight)
        #     torch.nn.init.zeros_(self.conv.conv5.bias)
        # if isinstance(self.conv.bn5, nn.BatchNorm2d):
        #     torch.nn.init.normal_(self.conv.bn5.weight.data, mean=1, std=0.02)
        #     torch.nn.init.constant_(self.conv.bn5.bias.data, 0)



class Node:
    def __init__(self, prior, to_play, action):
        self.visit_count = 0
        self.to_play = to_play
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.state = None
        self.action = action

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    # def select_child(self,game):
    #     """
    #     Select the child with the highest UCB score.
    #     """
    #     best_score = -np.inf
    #     best_action = -1
    #     best_child = None
    #
    #     for action,child in enumerate(self.children.items()):
    #         score = ucb_score(self, child[1])
    #         if not isFinish(self.state,game):
    #             if score> best_score:
    #                 best_score = score
    #                 best_child = child[1]
    #                 best_action=child[0]
    #
    #     return best_action,best_child

    # def ucb_score( self,child):
    #     """
    #     The score for an action that would transition between the parent and child.
    #     """
    #     prior_score = child.prior * math.sqrt(self.visit_count) / (child.visit_count + 1)
    #     if (child.visit_count > 0) and (child.prior > 0):
    #         # The value of the child is from the perspective of the opposing player
    #         value_score = -child.value()
    #     else:
    #         value_score = 0
    #
    #     return value_score + prior_score
    # def select_child(self, game):
    #     """
    #     Select the child with the highest UCB score.
    #     """
    #     if not self.children:
    #         return None
    #
    #     exploration_weight = 1.4  # tweakable parameter
    #     best_action, best_child = max(self.children.items(),
    #                                   key=lambda item: self.ucb_score(item[1]) + exploration_weight * math.sqrt(
    #                                       self.visit_count) * item[1].prior)
    #
    #     return best_action, best_child
    def ucb_score(parent, child):
        """
        The score for an action that would transition between the parent and child.
        """
        prior_score = child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1)
        if (child.visit_count > 0) and (child.prior > 0):
            # The value of the child is from the perspective of the opposing player
            value_score = -child.value()
        else:
            value_score = 0

        return value_score + prior_score

    def select_child(self, game):
        """
        Select the child with the highest UCB score.
        """
        if not self.children:
            return None

        exploration_weight = 1.4  # tweakable parameter
        best_action, best_child = max(self.children.items(),
                                      key=lambda item: self.ucb_score(item[1]) + exploration_weight * math.sqrt(
                                          self.visit_count) * item[1].prior)

        return best_action, best_child

    def expand(self, state, to_play, action_probs):
        """
        We expand a node and keep track of the prior policy probability given by neural network choice 5 first
        """
        self.to_play = to_play
        self.state = state
        indiceStateChildren = torch.sort(action_probs, descending=True).indices

        for a, prob in enumerate(action_probs):
            if prob > 0.0001:
                self.children[a] = Node(prior=prob.item(), to_play=self.to_play * -1,
                                        action=indiceStateChildren[a].item())

    def __repr__(self):
        """
        Debugger pretty print node info
        """
        prior = "{0:.2f}".format(self.prior)
        return "{} Prior: {} Count: {} Value: {}".format(self.state.__str__(), prior, self.visit_count, self.value())

    def actionCouldWin(self, state, param):
        return not isFinish(state)

    def copy(self):
        node = Node(self.prior, self.to_play, self.action)
        node.visit_count = self.visit_count
        node.state = self.state
        node.children = self.children.copy()
        return node


# def convertToBoard(state, racks):
#     nextBoard=np.copy(state)
#     i=0
#     for rack in racks:
#         nextBoard[12+rack[0]][0,i]=1
#         nextBoard[18+rack[1]][0,i]=1
#         i+=1
#     return nextBoard
def convertToBoard(state, racks):
    nextBoard = np.zeros_like(state)
    nextBoard[:12, :] = state[:12, :]
    for rack in racks:
        nextBoard[12 + rack[0], 0:len(rack)] = 1
        nextBoard[18 + rack[1], 0:len(rack)] = 1
    return nextBoard


import datetime
import multiprocessing


class MCTS:

    def backpropagate(self, search_path, value, to_play):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        for node in reversed(search_path):
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1

    def run(self, state, to_play, action, gamesimul, numsimul, indexprocess):
        with torch.no_grad():
            root = Node(0, to_play, action)
            statesignal = torch.tensor(state, dtype=torch.float32)
            action_probs, value = cnn(statesignal)
            valid_moves = get_valid_moves(gamesimul)
            action_probs = action_probs * valid_moves  # mask invalid moves
            action_probs /= torch.sum(action_probs)

            root.expand(state, to_play, torch.squeeze(action_probs, 0))
            i = 0
            rootcopy = root.copy()
            for nodechildren in sorted(root.children.items(), key=lambda x: x[1].prior, reverse=True)[0:5]:
                if (i == indexprocess):
                    rootcopy.children = {}
                    rootcopy.children[nodechildren[0]] = nodechildren[1]
                    break
                i += 1
            root = rootcopy.copy()
            if len(root.children) > 0:

                # for i in range(777):

                i = 0
                while i < numsimul:
                    i += 1

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
                        statesignal = torch.tensor(gridAll, dtype=torch.float32)
                        action_probs, value = cnn(statesignal)

                        valid_moves = get_valid_moves(gamesimul)

                        action_probs = action_probs * valid_moves  # mask invalid moves

                        action_probs /= torch.sum(action_probs)

                        node.expand(next_state, parent.to_play * -1, torch.squeeze(action_probs, 0))

                    self.backpropagate(search_path, value, parent.to_play * -1)

                return root
            else:
                return root

    def run_mctsmulti(self, state, to_play, action, game, num_simul, num_processes):
        with torch.no_grad():
            root = Node(0, to_play, action)
            statesignal = torch.tensor(state, dtype=torch.float32)
            action_probs, value = cnn(statesignal)
            game_simul = game.__copy__()
            valid_moves = get_valid_moves(game_simul)
            action_probs = action_probs * valid_moves  # mask invalid moves
            action_probs /= torch.sum(action_probs)

            root.expand(state, to_play, torch.squeeze(action_probs, 0))
            if len(root.children) > 0:
                start_time = datetime.datetime.now().time().strftime('%H:%M:%S')
                args_list = []
                for nodechildren in sorted(root.children.items(), key=lambda x: x[1].prior, reverse=True)[0:5]:
                    rootcopy = root.copy()
                    rootcopy.children = {}
                    rootcopy.children[nodechildren[0]] = nodechildren[1]
                    args_list.append(rootcopy)
                # Parallelize the simulations
                with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
                    with multiprocessing.Pool(processes=8) as pool:
                        results = list(pool.map(self.run_mcts, [args_list[0], 1, 0, game_simul, num_simul] * 8))
                end_time = datetime.datetime.now().time().strftime('%H:%M:%S')
                total_time = (datetime.datetime.strptime(end_time, '%H:%M:%S') - datetime.datetime.strptime(start_time,
                                                                                                            '%H:%M:%S'))
                print('total_time: ' + str(total_time))
            return root

    def run_mcts(self, root, to_play, action, gamesimul, numsimul):
        i = 0
        start_time = datetime.datetime.now().time().strftime('%H:%M:%S')
        while i < numsimul:
            i += 1

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
                statesignal = torch.tensor(gridAll, dtype=torch.float32)
                action_probs, value = cnn(statesignal)

                valid_moves = get_valid_moves(gamesimul)

                action_probs = action_probs * valid_moves  # mask invalid moves

                action_probs /= torch.sum(action_probs)

                node.expand(next_state, parent.to_play * -1, torch.squeeze(action_probs, 0))

            self.backpropagate(search_path, value, parent.to_play * -1)
        end_time = datetime.datetime.now().time().strftime('%H:%M:%S')
        total_time = (datetime.datetime.strptime(end_time, '%H:%M:%S') - datetime.datetime.strptime(start_time,
                                                                                                    '%H:%M:%S'))
        print('total_time:' + str(total_time))
        return root

    def runmulti(self, state, to_play, action, game, numsimul):
        with torch.no_grad():
            root = Node(0, to_play, action)
            statesignal = torch.tensor(state, dtype=torch.float32)
            action_probs, value = cnn(statesignal)
            gamesimul = game.__copy__()
            valid_moves = get_valid_moves(gamesimul)
            action_probs = action_probs * valid_moves  # mask invalid moves
            action_probs /= torch.sum(action_probs)

            root.expand(state, to_play, torch.squeeze(action_probs, 0))
            if len(root.children) > 0:
                arg_list = []
                for nodechildren in sorted(root.children.items(), key=lambda x: x[1].prior, reverse=True)[0:5]:
                    arg_list.append([nodechildren[1], 1, 0, gamesimul, numsimul])
                with multiprocessing.Pool(processes=8) as pool:
                    pool.starmap(self.run_mcts, arg_list)
                    pool.close()  # no more tasks
                    pool.join()  # wrap up current tasks

            else:
                return root


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
class MCTS_eval:

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
            action_probs, value = cnn(statesignal)
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
                        action_probs, value = cnn(statesignal)

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

# cnn = ConnectNet().to(cuda0)
cnn = ConnectNet()

cnn.init_weights()

# cnn_iter1 = ConnectNet().to(cuda0)
cnn_iter1 = ConnectNetiter()
cnn_iter1.init_weights()
n_step = 0


Step = namedtuple('Step', ['state', 'action', 'reward'])
from dataclasses import dataclass

mcts = MCTS()
mctseval = MCTS_eval()
mcts_iter = MCTS_iter()


@dataclass
class Step:
    state: []
    action: []
    reward: int = 0


global memory

global rewards
rewards = []

lossreward = nn.MSELoss()
# lossreward = nn.BCEWithLogitsLoss()
loss = nn.BCEWithLogitsLoss()
# optimizer = torch.optim.SGD(cnn.parameters(),lr=0.1,momentum=0.9,weight_decay=5e-4)
# optimizer = torch.optim.SGD(cnn.parameters(), lr=0.01, momentum=0.9)
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.0001)
optimizeriter = torch.optim.Adam(cnn_iter1.parameters(), lr=0.0001)
reward = 0.0
global gridnorme


def treeSearch(batch):
    inputs = []
    targets = []
    values = []
    for series in batch:
        inputs.append(series.state)
        targets.append(series.action)
        values.append(series.reward)
    return torch.tensor([t.numpy() for t in inputs], dtype=torch.float), torch.tensor([t.numpy() for t in targets],
                                                                                      dtype=torch.float), torch.tensor(
        [t.numpy() for t in values], dtype=torch.float)


def sample_batch(batch_size):  # creates an iterator that returns random batches
    ofs = 0
    vals = list(memory)
    np.random.shuffle(vals)
    while (ofs + 1) * batch_size <= len(memory):
        yield vals[ofs * batch_size:(ofs + 1) * batch_size]
        ofs += 1


epoch = 0


def contains(subseq, inseq):
    return any(inseq[pos:pos + len(subseq)] == subseq for pos in range(0, len(inseq) - len(subseq) + 1))


winner = 0


def isFinish(gridnorme, game):
    matrix = np.array(gridnorme)
    # Count occurrence of element '3' in each column
    count = np.count_nonzero(matrix == 1)
    return game.bag.isEmpty() and (len(game.player1.rack) == 0 or len(game.player2.rack) == 0)


epoch = 0

memory = deque()

maxWin = 10


def moyenne_glissante(valeurs, intervalle):
    indice_debut = (intervalle - 1) // 2
    liste_moyennes = [sum(valeurs[i - indice_debut:i + indice_debut + 1]) / intervalle for i in
                      range(indice_debut, len(valeurs) - indice_debut)]
    return liste_moyennes


class AlphaLoss(torch.nn.Module):
    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, y_value, value, y_policy, policy):
        value_error = (value - y_value) ** 2
        policy_error = torch.sum((-policy *
                                  (1e-8 + y_policy.float()).float().log()), 1)
        # total_error = (value_error.view(-1).float() + policy_error).mean()
        total_error = (value_error + policy_error.unsqueeze(1)).mean()
        return total_error



global moyenneWinBlue, BATCH_SIZE, pi_losses, v_losses, acurracy, runningloss
alphaloss = AlphaLoss()
BATCH_SIZE = 16
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()
plt.figure()
from IPython.display import clear_output

moyenneWinBlue = []
runningloss = []
acurracy = []


def savebraindequeZero():
    import pickle
    global memory
    memory = []
    pickle.dump(memory, open('buffer.pkl', 'wb'))
    print("=> saving memory zero... ")


def savebraindeque():
    import pickle
    global memory
    end_time = datetime.datetime.now().isoformat(' ')
    with open('buffer' + end_time + '.pkl', 'wb') as f:
        pickle.dump(memory, f)
    f.close()
    print("=> saving brainqueue... ")


def loadraindeque():
    import pickle
    global memory
    memory = []
    import os, glob
    for filename in glob.glob('buffer*.pkl'):
        with open(os.path.join(os.getcwd(), filename), 'rb') as f:
            tmp = pickle.load(f)
            memory.extend(tmp)
    f.close()

    print("=> saving brainqueue... ")


from torch.utils.data import Dataset


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, raw_data):
        self.data = []
        for i in range(len(raw_data)):
            board, pi, v = raw_data[i]
            board = torch.tensor(board, dtype=torch.float32)
            pi = torch.tensor(pi, dtype=torch.float32)
            v = torch.tensor(v, dtype=torch.float32)
            self.data.append((board, pi, v))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def loadraindequeTrain():
    import pickle
    global memory, training_loader
    memory = []
    import os, glob
    for filename in glob.glob('buffer*.pkl'):
        with open(os.path.join(os.getcwd(), filename), 'rb') as f:
            tmp = pickle.load(f)
            memory.extend(tmp)
    f.close()

    # create a dataset and dataloader
    custom_dataset = CustomDataset(memory)
    training_loader = torch.utils.data.DataLoader(custom_dataset, batch_size=4, shuffle=True, num_workers=2)
    print("=> saving brainqueue... ")


def loadraindeque2():
    import pickle
    global memory
    tmp = pickle.load(open('buffer2.pkl', 'rb'))
    memory = pickle.load(open('buffer.pkl', 'rb'))
    memory.extend(tmp)
    print("=> saving brainqueue2... ")


def localtrain():
    global maxWin, moyenneWinBlue, BATCH_SIZE, pi_losses, v_losses, acurracy, runningloss

    cnn = ConnectNet()
    cnn.init_weights()
    optimizer = torch.optim.SGD(cnn.parameters(), lr=0.05)
    # optimizer = torch.optim.Adam(cnn.parameters(), lr=0.00001)
    batch_idx = 0
    pi_losses = []
    v_losses = []

    while batch_idx < int(len(memory) / BATCH_SIZE):

        sample_ids = np.random.randint(0, len(memory), BATCH_SIZE)
        boards, pis, vs = list(zip(*[(memory[i]) for i in sample_ids]))
        # boards = torch.FloatTensor(np.stack(boards,axis=0))
        # boards = torch.FloatTensor(boards).cuda()

        boardsAll = []
        for board in boards:
            last_signal = torch.tensor(board, dtype=torch.float32)
            # last_signal = torch.FloatTensor(gridAll).cuda()

            boardsAll.append(last_signal.reshape(26, 54, 54))
        pisAll = []
        for policy in pis:
            # policy = torch.tensor(policy, dtype=torch.float32).cuda()

            valid_moves = np.zeros(23436)
            valid_moves[policy] = 1
            policy = torch.tensor(valid_moves, dtype=torch.float32)
            pisAll.append(policy)
        vsAll = []
        for value in vs:
            value = torch.tensor(value, dtype=torch.float32)
            # value = torch.tensor(value, dtype=torch.float32).cuda()
            vsAll.append(value)

        statesignal = torch.FloatTensor([t.detach().cpu().numpy() for t in boardsAll])
        # statesignal = torch.FloatTensor([t.detach().cpu().numpy() for t in boardsAll]).cuda()

        # target_pis =torch.FloatTensor([t.cpu().numpy() for t in pisAll]).cuda()
        target_pis = torch.FloatTensor([t.detach().cpu().numpy() for t in pisAll])
        # target_vs = torch.FloatTensor(vsAll).cuda().reshape(BATCH_SIZE,1)
        target_vs = torch.FloatTensor(vsAll).reshape(BATCH_SIZE, 1)

        # set_to_none=True here can modestly improve performance
        # with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        out_pi, out_v = cnn.forward(statesignal)
        total_error = alphaloss(out_v, target_vs, out_pi, target_pis)

        optimizer.zero_grad()
        total_error.backward()

        optimizer.step()
        pi_losses.append(float(total_error))

        batch_idx += 1
        print('Policy Loss:{:.2f}'.format(np.mean(pi_losses)))

        runningloss.append(np.mean(pi_losses))

    clear_output(wait=True)
    pal = sns.dark_palette('purple', 2)
    ax = sns.lineplot(data=runningloss, palette=pal, color='red', alpha=.5, linewidth=2)
    ax.legend(['loss'])
    # Customise some display properties
    ax.set_title('winblueloss')
    ax.set_ylabel('%')
    ax.set_xlabel(None)
    ax.set(ylim=(0, max(runningloss)))
    # Ask Matplotlib to show it
    plt.show()
    savebraintrain()
    savebrain1()


def getRack1From(tile):
    for index, rack in enumerate(game.player1.tilecolor):
        if rack == tile:
            shape = game.player1.tileshape[index]
            color = game.player1.tilecolor[index]
            # game.player1.delRack(color, shape)
            return [color, shape, 0, 0]


def getRack2From(tile):
    for index, rack in enumerate(game.player2.tilecolor):
        if rack == tile:
            shape = game.player2.tileshape[index]
            color = game.player2.tilecolor[index]
            # game.player2.delRack(color, shape)
            return [color, shape, 0, 0]


def getRack1FromShape(tile):
    for index, rack in enumerate(game.player1.tileshape):
        if rack == tile:
            shape = game.player1.tileshape[index]
            color = game.player1.tilecolor[index]
            # game.player1.delRack(color,shape)
            return [color, shape, 0, 0]


def getRack2FromShape(tile):
    for index, rack in enumerate(game.player2.tileshape):
        if rack == tile:
            shape = game.player2.tileshape[index]
            color = game.player2.tilecolor[index]
            # game.player2.delRack(color, shape)
            return [color, shape, 0, 0]


def convertToRealTiles(boardPlay, game):
    for tiles in game.listValidMoves:
        if boardPlay == tuple([[tile[0], tile[1]] for tile in tiles]):
            return tiles
    else:
        return []


def deepGridCopy(gridnorme):
    val = np.zeros(shape=(26, 54, 54))
    for i, grid in enumerate(gridnorme):
        for x, lig in enumerate(grid):
            for y, col in enumerate(lig):
                val[i][x][y] = col

    return val


def local(num_game):
    global gridnorme, game, epoch, memory, history, n_steps, maxWin

    import datetime
    import pickle
    maxWin = [0, 0]
    for h in range(0, num_game):
        n_steps = []
        game = newGame.GameNumpy()
        # game.setActionprobtest()
        game.actionprob = pickle.load(open('gameActionProb.pkl', 'rb'))
        gridnorme = np.zeros(shape=(26, 54, 54))

        first = random.choice([True, False])

        start_time = datetime.datetime.now().time().strftime('%H:%M:%S')
        while not ((game.player1.rackCount() == 0 or (
                game.player2.rackCount() == 0) and game.bag.bagCount() == 0)) and game.test3round():

            if first:
                game.listValidMovePlayer1()
                if len(game.listValidMoves) > 0:
                    gridnorme = convertToBoard(gridnorme, game.player1.getRack())

                    actions = mcts.run(gridnorme, 1, 0, game.__copy__(), 600,1)
                    # actions = torch.tensor([actions.children[visit].visit_count for visit in actions.children],
                    #                        dtype=torch.float32).cuda()
                    childvisitcount = torch.tensor(
                        [[actions.children[visit].visit_count] for visit in actions.children],
                        dtype=torch.float32)
                    childvisitcount /= torch.sum(childvisitcount)
                    # game.setActionprob()
                    actionPosition = list(actions.children.keys())[
                        torch.sort(childvisitcount, descending=True).indices[0]]
                    boardPlay = game.actionprob[actionPosition]

                    boardPlay = convertToRealTiles(boardPlay, game)

                    for tile in boardPlay:
                        if game.place(tile[0], tile[1], tile[2], tile[3]):
                            game.player1.delRack(tile[0], tile[1])
                        else:
                            game.round += 1

                    if (game.player2.rackCount() + game.player1.rackCount() + game.bag.bagCount() + len(
                            np.where(game.tilecolor != 0)[0]) != 108):
                        print('\rbaordplay1:{0}'.format(boardPlay))

                    game.player1.addTileToRack(game.bag)

                    game.round = 0

                    print('\rbag:{0}'.format(game.bag.bagCount()))
                    print('\rbpordplay1:{0}'.format(boardPlay))

                    game.player1.point += game.getpoint([[x[2], x[3]] for x in boardPlay])
                    print('\rpoint:{0}'.format(game.player1.point))
                    n_steps.append([deepGridCopy(gridnorme), actionPosition, 0])

                else:
                    game.player1.newRack(game.bag)
                    game.round += 1
                    n_steps.append([deepGridCopy(gridnorme), 0, 0])

                first = not first


            else:
                game.listValidMovePlayer2()
                if len(game.listValidMoves) > 0:
                    gridnorme = convertToBoard(gridnorme, game.player2.getRack())

                    actions = mcts_iter.run(gridnorme, -1, 0, game)
                    # actions = torch.tensor([actions.children[visit].visit_count for visit in actions.children],
                    #                        dtype=torch.float32).cuda()
                    childvisitcount = torch.tensor(
                        [[actions.children[visit].visit_count] for visit in actions.children],
                        dtype=torch.float32)
                    childvisitcount /= torch.sum(childvisitcount)
                    # game.setActionprob()
                    actionPosition = list(actions.children.keys())[
                        torch.sort(childvisitcount, descending=True).indices[0]]
                    boardPlay = game.actionprob[actionPosition]

                    boardPlay = convertToRealTiles(boardPlay, game)

                    for tile in boardPlay:
                        if game.place(tile[0], tile[1], tile[2], tile[3]):
                            game.player2.delRack(tile[0], tile[1])

                        else:
                            game.round += 1

                    if (game.player2.rackCount() + game.player1.rackCount() + game.bag.bagCount() + len(
                            np.where(game.tilecolor != 0)[0]) != 108):
                        print('\rbaordplay2:{0}'.format(boardPlay))
                    game.player2.addTileToRack(game.bag)

                    game.round = 0

                    print('\rbag:{0}'.format(game.bag.bagCount()))
                    print('\rbpordplay2:{0}'.format(boardPlay))

                    game.player2.point += game.getpoint([[x[2], x[3]] for x in boardPlay])
                    print('\rpoint:{0}'.format(game.player2.point))
                    n_steps.append([deepGridCopy(gridnorme), actionPosition, 0])

                else:
                    game.player2.newRack(game.bag)
                    game.round += 1
                    n_steps.append([deepGridCopy(gridnorme), 0, 0])

                first = not first

        end_time = datetime.datetime.now().time().strftime('%H:%M:%S')
        total_time = (datetime.datetime.strptime(end_time, '%H:%M:%S') - datetime.datetime.strptime(start_time,
                                                                                                    '%H:%M:%S'))
        print('total_time:' + str(total_time))

        if game.winner() == 1:
            maxWin[0] += 1

            reward = 1
            for rew in range(len(n_steps) - 1, 0, -1):
                n_steps[rew][2] = reward
                reward = -reward
        else:
            maxWin[1] += 1
            reward = 1
            for rew in range(len(n_steps) - 1, 0, -1):
                n_steps[rew][2] = reward
                reward = -reward
        # else:
        #     for rew in range(0, len(n_steps)):
        #         n_steps[rew][2] = 0

        memory.extend(n_steps)

        n_steps = []
    savebraindeque()


def localevaluation(num_game):
    global gridnorme, game, epoch, memory, history, n_steps, maxWin

    import datetime
    import pickle
    maxWin = [0, 0]
    for h in range(0, num_game):
        n_steps = []
        game = newGame.GameNumpy()
        # game.setActionprobtest()
        game.actionprob = pickle.load(open('gameActionProb.pkl', 'rb'))
        gridnorme = np.zeros(shape=(26, 54, 54))

        first = random.choice([True, False])

        start_time = datetime.datetime.now().time().strftime('%H:%M:%S')
        while not ((game.player1.rackCount() == 0 or (
                game.player2.rackCount() == 0) and game.bag.bagCount() == 0)) and game.test3round():

            if first:
                game.listValidMovePlayer1All()
                if len(game.listValidMoves) > 0:
                    gridnorme = convertToBoard(gridnorme, game.player1.getRack())

                    actions = mctseval.run(gridnorme, 1, 0, game.__copy__())
                    # actions = torch.tensor([actions.children[visit].visit_count for visit in actions.children],
                    #                        dtype=torch.float32).cuda()
                    childvisitcount = torch.tensor(
                        [[actions.children[visit].visit_count] for visit in actions.children],
                        dtype=torch.float32)
                    childvisitcount /= torch.sum(childvisitcount)
                    # game.setActionprob()
                    actionPosition = list(actions.children.keys())[
                        torch.sort(childvisitcount, descending=True).indices[0]]
                    boardPlay = game.actionprob[actionPosition]

                    boardPlay = convertToRealTiles(boardPlay, game)

                    for tile in boardPlay:
                        if game.place(tile[0], tile[1], tile[2], tile[3]):
                            game.player1.delRack(tile[0], tile[1])
                        else:
                            game.round += 1

                    if (game.player2.rackCount() + game.player1.rackCount() + game.bag.bagCount() + len(
                            np.where(game.tilecolor != 0)[0]) != 108):
                        print('\rbaordplay1:{0}'.format(boardPlay))

                    game.player1.addTileToRack(game.bag)

                    game.round = 0

                    print('\rbag:{0}'.format(game.bag.bagCount()))
                    print('\rbpordplay1:{0}'.format(boardPlay))

                    game.player1.point += game.getpoint([[x[2], x[3]] for x in boardPlay])
                    print('\rpoint:{0}'.format(game.player1.point))

                else:
                    game.player1.newRack(game.bag)
                    game.round += 1

                first = not first


            else:
                game.listValidMovePlayer2All()
                if len(game.listValidMoves) > 0:
                    gridnorme = convertToBoard(gridnorme, game.player2.getRack())

                    actions = mcts_iter.run(gridnorme, -1, 0, game)
                    # actions = torch.tensor([actions.children[visit].visit_count for visit in actions.children],
                    #                        dtype=torch.float32).cuda()
                    childvisitcount = torch.tensor(
                        [[actions.children[visit].visit_count] for visit in actions.children],
                        dtype=torch.float32)
                    childvisitcount /= torch.sum(childvisitcount)
                    # game.setActionprob()
                    actionPosition = list(actions.children.keys())[
                        torch.sort(childvisitcount, descending=True).indices[0]]
                    boardPlay = game.actionprob[actionPosition]

                    boardPlay = convertToRealTiles(boardPlay, game)

                    for tile in boardPlay:
                        if game.place(tile[0], tile[1], tile[2], tile[3]):
                            game.player2.delRack(tile[0], tile[1])

                        else:
                            game.round += 1

                    if (game.player2.rackCount() + game.player1.rackCount() + game.bag.bagCount() + len(
                            np.where(game.tilecolor != 0)[0]) != 108):
                        print('\rbaordplay2:{0}'.format(boardPlay))
                    game.player2.addTileToRack(game.bag)

                    game.round = 0

                    print('\rbag:{0}'.format(game.bag.bagCount()))
                    print('\rbpordplay2:{0}'.format(boardPlay))

                    game.player2.point += game.getpoint([[x[2], x[3]] for x in boardPlay])
                    print('\rpoint:{0}'.format(game.player2.point))

                else:
                    game.player2.newRack(game.bag)
                    game.round += 1

                first = not first

        end_time = datetime.datetime.now().time().strftime('%H:%M:%S')
        total_time = (datetime.datetime.strptime(end_time, '%H:%M:%S') - datetime.datetime.strptime(start_time,
                                                                                                    '%H:%M:%S'))
        print('total_time:' + str(total_time))

        if game.winner() == 1:
            maxWin[0] += 1


        else:
            maxWin[1] += 1


import csv


def getmaxWin():
    global maxWin
    return maxWin


def savebrain1():
    global savefile
    global cnn, optimizer

    print("=> saving checkpoint... ")
    checkpoint = {'model': cnn,
                  'state_dict': cnn.state_dict(),
                  'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, 'bestrandom.pth')

    print("=> saving checkpoint... ")


def savebrainmultiprocess(cnn, optimizer):
    global savefile

    print("=> saving checkpoint... ")
    checkpoint = {'model': cnn,
                  'state_dict': cnn.state_dict(),
                  'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, 'bestrandom.pth')

    print("=> saving checkpoint... ")


def savebraintrain():
    global savefile
    global runningloss

    with open('loss_2000.csv', mode='w') as loss_file:
        blue_writer = csv.writer(loss_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        blue_writer.writerow(runningloss)

    print("=> saving moyenne... ", runningloss[len(runningloss) - 1])


def savegameboard(boardplay):
    global savefile

    with open('game_test.csv', mode='a') as loss_file:
        blue_writer = csv.writer(loss_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        blue_writer.writerow(boardplay)

    print("=> saving game... ")


def loadbrain2():
    global cnn_iter1, optimizeriter
    cnn_iter1 = ConnectNetiter()
    cnn_iter1.init_weights()
    if os.path.isfile('bestrandomiter.pth'):
        print("=> loading checkpoint... ")
        # checkpoint = torch.load('bestrandom.pth', map_location=cuda0)
        checkpoint = torch.load('bestrandomiter.pth')
        cnn_iter1.load_state_dict(checkpoint['state_dict'])
        optimizeriter.load_state_dict(checkpoint['optimizer'])

        print("done !")
    else:
        print("no checkpoint found...")


def loadcsv():
    global cnn, optimizer, rewardcnn, cnnred, rewardcnnred, runningloss

    if os.path.isfile('loss_2000.csv'):

        with open('loss_2000.csv', newline='') as f:
            reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
            runningloss = list(reader)[0]

        print("=> loading checkpoint... ")


    else:
        print("no checkpoint found...")


def loadbrain1():
    global cnn, optimizer

    # cnn = ConnectNet().to(cuda0)
    cnn = ConnectNet()
    cnn.init_weights()
    # cnn_iter1 = ConnectNet().to(cuda0)
    if os.path.isfile('bestrandom.pth'):
        print("=> loading checkpoint... ")
        # checkpoint = torch.load('bestrandom.pth', map_location=cuda0)
        checkpoint = torch.load('bestrandom.pth')
        cnn.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("done !")
    else:
        print("no checkpoint found...")
