


import os
from copy import deepcopy

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import numpy as np
from collections import namedtuple, deque
import torch

import Bag as newGame




cuda0 = torch.device('cuda:0')




global epoch
epoch = 0
import json
import random
import math
import logging

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)




def get_next_state(board, player, action):
    nextBoard = np.copy(board)
    lign = 0
    while nextBoard[lign][action] != 0 and sum(
            (x != 0) for x in nextBoard.transpose().flatten()) < 42:
        lign = (lign + 1) % 6
    nextBoard[lign][action] = player

    # Return the new game, but
    # change the perspective of the game with negative
    return nextBoard, -player


def has_legal_moves(board):
    for colonne in range(7):
        if not colonnepleine(board, colonne) and sum(
                (x != 0) for x in board.transpose().flatten()) < 42:
            return True
    return False


def get_valid_moves(board):
    # All moves are invalid by default
    game.setActionprob()
    valid_moves=np.zeros(len(game.actionprob))
    probscolor=[]
    probsshape=[]

    for index,i in enumerate(game.listValidMoves):
        
        if len(i)<2:
            for tile in i:
                probscolor.append([newGame.TileColor[tile.color],0,0,tile.coordinate.x,tile.coordinate.y])
                probsshape.append([0,newGame.TileShape[tile.shape],0,tile.coordinate.x,tile.coordinate.y])

        else:
            if (i[0].coordinate.x-i[1].coordinate.x)>0:
                for tile in i:
                    probscolor.append([newGame.TileColor[tile.color],0,0,i[0].coordinate.x,i[0].coordinate.y])
                    probsshape.append([0,newGame.TileShape[tile.shape],0,i[0].coordinate.x,i[0].coordinate.y])
                break        
            if (i[0].coordinate.x-i[1].coordinate.x)<0:
                for tile in i:
                    probscolor.append([newGame.TileColor[tile.color],0,1,i[0].coordinate.x,i[0].coordinate.y])
                    probsshape.append([0,newGame.TileShape[tile.shape],1,i[0].coordinate.x,i[0].coordinate.y])
                break    
            if (i[0].coordinate.y-i[1].coordinate.y)>0:
                for tile in i:
                    probscolor.append([newGame.TileColor[tile.color],0,2,i[0].coordinate.x,i[0].coordinate.y])
                    probsshape.append([0,newGame.TileShape[tile.shape],2,i[0].coordinate.x,i[0].coordinate.y])
                break    
            if (i[0].coordinate.y-i[1].coordinate.y)<0:
                for tile in i:
                    probscolor.append([newGame.TileColor[tile.color],0,3,i[0].coordinate.x,i[0].coordinate.y])
                    probsshape.append([0,newGame.TileShape[tile.shape],3,i[0].coordinate.x,i[0].coordinate.y])
                break                            

    for prob in probscolor:
        valid_moves[game.actionprob.index([prob])]=1


    for prob in probsshape:
        valid_moves[game.actionprob.index([prob])]=1    
         
    

    return valid_moves


def get_reward_for_player(board, player):
    # return None if not ended, 1 if player 1 wins, -1 if player 1 lost

    if colWin(board) or ligneWin(board) or diagRigthToLeftWin(board) or diagLeftToRightWin(board):
        if winner == 1:
            return 1
        else:
            return -1
    # if has_legal_moves(board):
    #     return None

    return 0


def get_canonical_board(board, player):
    return player * board


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


# In[245]:


def add_dirichlet_noise(child_priors):
        dirichlet_input = [0.1 for x in range(len(child_priors))]
        dirichlet_list = np.random.dirichlet(dirichlet_input)
        noisy_psa_vector = []
        for idx, psa in enumerate(child_priors):
            noisy_psa_vector.append(
                (1 - 0.25) * psa + 0.25 * dirichlet_list[idx])
        child_priors =noisy_psa_vector[0]


        return (child_priors)





class Node:
    def __init__(self, prior, to_play):
        self.visit_count = 0
        self.to_play = to_play
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.state = None

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def select_child(self):
        """
        Select the child with the highest UCB score.
        """
        best_score = -np.inf
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            score = ucb_score(self, child)
            if not colonnepleine(self.state, action):
                if score> best_score:
                    best_score = score
                    best_action = action
                    best_child = child

        return best_action, best_child

    def expand(self, state, to_play, action_probs):
        """
        We expand a node and keep track of the prior policy probability given by neural network choice 5 first
        """
        self.to_play = to_play
        self.state = state
        val=torch.topk(action_probs, 5).values
        for i in range(len(action_probs)):
          if action_probs[i] in val and self.actionCouldWin(self.state, action_probs[i]):
            self.children[i] = Node(prior=action_probs[i].item(), to_play=self.to_play * -1)
          else:
            self.children[i] = Node(prior=0, to_play=self.to_play * -1)



    def __repr__(self):
        """
        Debugger pretty print node info
        """
        prior = "{0:.2f}".format(self.prior)
        return "{} Prior: {} Count: {} Value: {}".format(self.state.__str__(), prior, self.visit_count, self.value())

    def actionCouldWin(self, state, param):
        return not isFinish(state)


class MCTS:

    def backpropagate(self, search_path, value, to_play):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        for node in reversed(search_path):
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1

    def run(self, state, to_play):
      with torch.no_grad():      
        #   gridnormeOne = np.zeros(shape=(26,54,54))
        #   gridnormenegOne = np.zeros(shape=(6, 7))
        #   gridnormeZero = np.zeros(shape=(6, 7))
          root = Node(0, to_play)
        #   convert_zero(state, gridnormeZero)
        #   convert_one(state, gridnormeOne)
        #   convert_neg_one(state, gridnormenegOne)
          gridAll = state
          #statesignal = torch.tensor([gridAll], dtype=torch.float32).cuda()
          statesignal = torch.tensor([gridAll], dtype=torch.float32)
          action_probs, value = cnn(statesignal)
          
          

          valid_moves = get_valid_moves(state)

          #action_probs = action_probs * torch.tensor([valid_moves], dtype=torch.float32).cuda()
          action_probs = action_probs * torch.tensor(valid_moves, dtype=torch.float32)  # mask invalid moves
          action_probs /= torch.sum(action_probs)
          action_probs = add_dirichlet_noise(action_probs)
          root.expand(state, to_play, action_probs)

          #for i in range(777):
          for i in range(5):    
              
              node = root
              search_path = [node]
              print('\rsimulation:{:.2f}'.format(i),end='')

              # SELECT
              while node.expanded():
                  action, node = node.select_child()
                  search_path.append(node)

              parent = search_path[-2]
              state = parent.state
              # Now we're at a leaf node and we would like to expand
              # Players always play from their own perspective
              next_state, _ = get_next_state(state, player=1, action=action)
              # Get the board from the perspective of the other player
              next_state = get_canonical_board(next_state, player=-1)

              # The value of the new state from the perspective of the other player
              value = get_reward_for_player(next_state, player=1)
              if value==0 and has_legal_moves(next_state):
                  # If the game has not ended:
                  # EXPAND
                  
                  gridAll = next_state
                  #statesignal = torch.tensor([gridAll], dtype=torch.float32).cuda()
                  statesignal = torch.tensor([gridAll], dtype=torch.float32)
                  statesignal = statesignal.reshape(3, 6, 7)
                  
                  action_probs, value = cnn(statesignal)
                  valid_moves = get_valid_moves(next_state)
                  #action_probs = action_probs * torch.tensor([valid_moves], dtype=torch.float32).cuda()
                  action_probs = action_probs * torch.tensor([valid_moves], dtype=torch.float32)  # mask invalid moves
                  action_probs /= torch.sum(action_probs)

                  node.expand(next_state, parent.to_play * -1, action_probs)

              self.backpropagate(search_path, value, parent.to_play * -1)
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

    def run(self, state, to_play):
      with torch.no_grad():      
          gridnormeOne = np.zeros(shape=(26,54,54))
          
          root = Node(0, to_play)
          
          #statesignal = torch.tensor([gridAll], dtype=torch.float32).cuda()
          statesignal = torch.tensor([gridAll], dtype=torch.float32)

          action_probs, value = cnn_iter1(statesignal)

          valid_moves = get_valid_moves(state)

          #action_probs = action_probs * torch.tensor([valid_moves], dtype=torch.float32).cuda() 
          action_probs = action_probs * torch.tensor([valid_moves], dtype=torch.float32) # mask invalid moves
          action_probs /= torch.sum(action_probs)
          action_probs = add_dirichlet_noise(action_probs)
          root.expand(state, to_play, action_probs)

          for i in range(777):
              
              node = root
              search_path = [node]
              print('\rsimulation:{:.2f}'.format(i),end='')

              # SELECT
              while node.expanded():
                  action, node = node.select_child()
                  search_path.append(node)

              parent = search_path[-2]
              state = parent.state
              # Now we're at a leaf node and we would like to expand
              # Players always play from their own perspective
              next_state, _ = get_next_state(state, player=1, action=action)
              # Get the board from the perspective of the other player
              next_state = get_canonical_board(next_state, player=-1)

              # The value of the new state from the perspective of the other player
              value = get_reward_for_player(next_state, player=1)
              if value==0 and has_legal_moves(next_state):
                  # If the game has not ended:
                  # EXPAND
                  
                  gridAll = next_state
                  #statesignal = torch.tensor([gridAll], dtype=torch.float32).cuda()
                  statesignal = torch.tensor([gridAll], dtype=torch.float32)

                  statesignal = statesignal.reshape(3, 6, 7)
                  
                  action_probs, value = cnn_iter1(statesignal)
                  valid_moves = get_valid_moves(next_state)
                  #action_probs = action_probs * torch.tensor([valid_moves], dtype=torch.float32).cuda() 
                  action_probs = action_probs * torch.tensor([valid_moves], dtype=torch.float32) # mask invalid moves
                  action_probs /= torch.sum(action_probs)

                  node.expand(next_state, parent.to_play * -1, action_probs)

              self.backpropagate(search_path, value, parent.to_play * -1)
          return root   




class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.action_size = 7
       # self.conv1 = nn.Conv2d(3, 1024, kernel_size=4, stride=1, padding=1).cuda()
        #self.bn1 = nn.BatchNorm2d(1024).cuda()
        #self.conv2 = nn.Conv2d(1024, 2048, kernel_size=4, stride=1, padding=1).cuda()
        #self.bn2 = nn.BatchNorm2d(2048).cuda()
        #self.conv3 = nn.Conv2d(2048, 4096, kernel_size=4, stride=1, padding=1).cuda()
        #self.bn3 = nn.BatchNorm2d(4096).cuda()
        #self.conv4 = nn.Conv2d(4096, 2048, kernel_size=4, stride=1, padding=1).cuda()
        #self.bn4 = nn.BatchNorm2d(2048).cuda()
        #self.conv5 = nn.Conv2d(2048, 512, kernel_size=4, stride=1, padding=1).cuda()
        #self.bn5 = nn.BatchNorm2d(512).cuda()
        self.conv1 = nn.Conv2d(26, 256, kernel_size=12, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=12, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        #self.conv3 = nn.Conv2d(2048, 4096, kernel_size=4, stride=1, padding=1)
        #self.bn3 = nn.BatchNorm2d(4096)
        #self.conv4 = nn.Conv2d(4096, 2048, kernel_size=4, stride=1, padding=1)
        #self.bn4 = nn.BatchNorm2d(2048)
        #self.conv5 = nn.Conv2d(2048, 512, kernel_size=4, stride=1, padding=1)
        #self.bn5 = nn.BatchNorm2d(512)


    def forward(self, s):
        s = s.view(-1,26, 54, 54)  # batch_size x channels x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))
        s = F.relu(self.bn2(self.conv2(s)))
        #s = F.relu(self.bn3(self.conv3(s)))
        #s = F.relu(self.bn4(self.conv4(s)))
        #s = F.relu(self.bn5(self.conv5(s)))

        return s


class ResBlock(nn.Module):
    def __init__(self, inplanes=512, planes=512, stride=1, downsample=None):
        super(ResBlock, self).__init__()
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
        out = F.relu(self.bn1(out))
        out = self.drp(out)
        out = self.conv2(out)
        out = F.relu(self.bn2(out))
        out = self.drp(out)
        out += residual
        out = F.relu(out)
        return out


class OutBlock(nn.Module):
    # shape=6*7*32
    shape = 663552

    def __init__(self):
        super(OutBlock, self).__init__()

        self.fc1 = nn.Linear(self.shape,512)
        self.fc2 = nn.Linear(512, 1)
        self.drp = nn.Dropout(0.3)



        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(512, 460800)

    def forward(self, s):
        v = s.view(-1, self.shape)  # batch_size X channel X height X width
        v = self.drp(F.relu(self.fc1(v)))
        v = torch.tanh(self.fc2(v))


        p = s.view(-1, self.shape)
        p = self.drp(F.relu(self.fc1(p)))
        p = self.fc(p)
        p = self.logsoftmax(p).exp()
        return p, v


class ConnectNet(nn.Module):
    def __init__(self):
        super(ConnectNet, self).__init__()
        #self.conv = ConvBlock().cuda()
        self.conv = ConvBlock()
        for block in range(30):
            #setattr(self, "res_%i" % block, ResBlock().cuda())
            setattr(self, "res_%i" % block, ResBlock())
        self.outblock = OutBlock()

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
        if isinstance(self.conv.conv1, nn.Conv2d):
            torch.nn.init.xavier_uniform_(self.conv.conv1.weight)
            torch.nn.init.zeros_(self.conv.conv1.bias)
        if isinstance(self.conv.bn1, nn.BatchNorm2d):
            torch.nn.init.normal_(self.conv.bn1.weight.data, mean=1, std=0.02)
            torch.nn.init.constant_(self.conv.bn1.bias.data, 0)
        if isinstance(self.conv.conv2, nn.Conv2d):
            torch.nn.init.xavier_uniform_(self.conv.conv2.weight)
            torch.nn.init.zeros_(self.conv.conv2.bias)
        if isinstance(self.conv.bn2, nn.BatchNorm2d):
            torch.nn.init.normal_(self.conv.bn2.weight.data, mean=1, std=0.02)
            torch.nn.init.constant_(self.conv.bn2.bias.data, 0)
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




#cnn = ConnectNet().to(cuda0)
cnn = ConnectNet()
cnn.init_weights()
#cnn_iter1 = ConnectNet().to(cuda0)
cnn_iter1 = ConnectNet()
cnn_iter1.init_weights()
n_step = 0

Step = namedtuple('Step', ['state', 'action', 'reward'])
from dataclasses import dataclass

mcts = MCTS()
mcts_iter=MCTS_iter()
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
# optimizer = torch.optim.SGD(cnn.parameters(),lr=0.01,momentum=0.9,weight_decay=5e-4)
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.0000001)
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





def isFinish(gridnorme):
    matrix =np.array(gridnorme)
# Count occurrence of element '3' in each column
    count = np.count_nonzero(matrix == 1)
    return game.bag.isEmpty()




epoch = 0

memory = deque()




maxWin = 10




import time




def moyenne_glissante(valeurs, intervalle):
    indice_debut = (intervalle - 1) // 2
    liste_moyennes = [sum(valeurs[i - indice_debut:i + indice_debut + 1]) / intervalle for i in range(indice_debut, len(valeurs) - indice_debut)]
    return liste_moyennes




class AlphaLoss(torch.nn.Module):
    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, y_value, value, y_policy, policy):
        value_error = (value - y_value) ** 2
        policy_error = torch.sum((-policy *
                                  (1e-8 + y_policy.float()).float().log()), 1)
        total_error = (value_error.view(-1).float() + policy_error).mean()
        return total_error




global moyenneWinBlue,BATCH_SIZE,pi_losses,v_losses,acurracy,runningloss
alphaloss = AlphaLoss()
BATCH_SIZE=64
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
plt.figure()
from IPython.display import clear_output
moyenneWinBlue=[]
runningloss = []
acurracy=[]
def localtrain():
    global maxWin,moyenneWinBlue,BATCH_SIZE,pi_losses,v_losses,acurracy,runningloss
    
   
    
    

    for epoch in range(0, 20):


      batch_idx = 0
      
      pi_losses = []
      v_losses = []

      while batch_idx < int(len(memory) / BATCH_SIZE):

        sample_ids = np.random.randint(0, len(memory), BATCH_SIZE)
        boards, pis, vs = list(zip(*[(memory[i]) for i in sample_ids]))
        gridnormeOne = np.zeros(shape=(6, 7))
        gridnormenegOne = np.zeros(shape=(6, 7))
        gridnormeZero = np.zeros(shape=(6, 7))
        boards = torch.FloatTensor(boards).cuda()

        boardsAll=[]
        for board in boards:
          
            gridAll = [gridnormeZero, gridnormeOne, gridnormenegOne]
            last_signal = torch.FloatTensor(gridAll).cuda()
            last_signal = last_signal.reshape(3, 6, 7)
            boardsAll.append(last_signal)
        pisAll = []
        for policy in pis:
            policy = torch.tensor(policy, dtype=torch.float32).cuda()
            pisAll.append(policy)
        vsAll = []
        for value in vs:
            value = torch.tensor(value, dtype=torch.float32).cuda()
            vsAll.append(value)

        statesignal = torch.FloatTensor([t.detach().cpu().numpy() for t in boardsAll]).cuda()

        target_pis =torch.FloatTensor([t.cpu().numpy() for t in pisAll]).cuda()
        target_vs = torch.FloatTensor(vsAll).cuda().reshape(BATCH_SIZE,1)
          
          
        
        out_pi, out_v = cnn(statesignal)
        total_error=alphaloss(out_v,target_vs,out_pi,target_pis)
        optimizer.zero_grad()
        pi_losses.append(float(total_error))
        total_error.backward(retain_graph=True)
        optimizer.step()

        batch_idx += 1
    print('Policy Loss:{:.2f}'.format(np.mean(pi_losses)))
    
    
    runningloss.append(np.mean(pi_losses))        
    
    clear_output(wait=True)     
    pal = sns.dark_palette('purple',2)
    ax = sns.lineplot(data=runningloss,palette=pal, color='red',  alpha=.5, linewidth=2)
    ax.legend(['loss'])
    # Customise some display properties
    ax.set_title('winblueloss')
    ax.set_ylabel('%')
    ax.set_xlabel(None)
    ax.set(ylim=(0, max(runningloss)))
    # Ask Matplotlib to show it
    plt.show()
    savebraintrain()
     




def local(num_game):
    

    global gridnorme, game, epoch, memory, history, n_steps, maxWin
    

    interval = []

    
    for h in range(0, num_game):
        start_time = time.time()
        n_steps = []
        gridnorme = np.zeros(shape=(26,54, 54))
        game=newGame.Game()
        
        
        first = random.choice([True, False])
        
        for i in range(0,54):
            

           

            if first:
                

                if not isFinish(gridnorme):
                    
                      game.permutePlayer1()  
                      actions = mcts.run(gridnorme, -1)
                      #actions = torch.tensor([actions.children[visit].visit_count for visit in actions.children],
                      #                        dtype=torch.float32).cuda()
                      actions = torch.tensor([actions.children[visit].visit_count for visit in actions.children],
                                              dtype=torch.float32)
                      actions /= torch.sum(actions)
                      colonnes = torch.sort(actions, descending=True)
                      action2rotation = [0, 1, 2, 3, 4, 5, 6]
                      colonne = action2rotation[colonnes.indices[0]]
                      j = 0
                      while colonnepleine(gridnorme, colonne) and sum(
                              (x != 0) for x in gridnorme.transpose().flatten()) < 42:
                          j = (j + 1) % len(colonnes.indices)
                          colonne = action2rotation[colonnes.indices[j]]

                      colonne = action2rotation[colonne]
                      lign = 0
                      while gridnorme[lign][colonne] != 0 and sum(
                              (x != 0) for x in gridnorme.transpose().flatten()) < 42:
                          lign = (lign + 1) % 6

                      n_steps.append([deepcopy(gridnorme), actions, 0])
                      gridnorme[lign][colonne] = -1
                      first = not first

                else:
                    break
            else:
                if not isFinish(gridnorme):

                      
                      
                      game.permutePlayer2()  
                      actions = mcts.run(gridnorme, 1)
                      #actions = torch.tensor([actions.children[visit].visit_count for visit in actions.children],
                      #                        dtype=torch.float32).cuda()
                      actions = torch.tensor([actions.children[visit].visit_count for visit in actions.children],
                                              dtype=torch.float32)                     
                      actions /= torch.sum(actions)
                      colonnes = torch.sort(actions, descending=True)
                      action2rotation = [0, 1, 2, 3, 4, 5, 6]
                      colonne = action2rotation[colonnes.indices[0]]
                      j = 0
                      while colonnepleine(gridnorme, colonne) and sum(
                              (x != 0) for x in gridnorme.transpose().flatten()) < 42:
                          j = (j + 1) % len(colonnes.indices)
                          colonne = action2rotation[colonnes.indices[j]]

                      colonne = action2rotation[colonne]
                      lign = 0
                      while gridnorme[lign][colonne] != 0 and sum(
                              (x != 0) for x in gridnorme.transpose().flatten()) < 42:
                          lign = (lign + 1) % 6

                      n_steps.append([deepcopy(gridnorme), actions, 0])
                      gridnorme[lign][colonne] = 1
                      first = not first
                else:
                    break
        
        n_steps.append([deepcopy(gridnorme), [0, 0, 0, 0, 0, 0, 0], 0])
        if winner == 1:

            reward=-1
            for rew in range(len(n_steps) - 1, 0, -1):
                n_steps[rew][2] = reward
                reward=-reward
        elif winner == -1:

            reward = 1
            for rew in range(len(n_steps) - 1, 0, -1):
                n_steps[rew][2] = reward
                reward=-reward


        else:
            for rew in range(0, len(n_steps)):
                n_steps[rew][2] = 0

        memory.extend(n_steps)

        n_steps = []

        
            
        interval.append(time.time() - start_time)
        if h > 5:
            localtrain()
        print('Total time in seconde:{:.2f}  '.format(np.mean(interval)),' ',h)
        if h%10==0:
          savebrain1()
        

    
    




import pickle
import csv
def savebrain1():
    global savefile
    global cnn, optimizer, cnnred
    

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

    print("=> saving moyenne... ",runningloss[len(runningloss)-1])
        
def loadbrain2():
    global cnn, optimizer, rewardcnn, cnnred, rewardcnnred
    if os.path.isfile('best_iter200.pth'):
        print("=> loading checkpoint... ")
       
        checkpoint = torch.load('best_iter200.pth', map_location=cuda0)
        cnn_iter1.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
       

        print("done !")
    else:
        print("no checkpoint found...")
        
def loadcsv():
    global cnn, optimizer, rewardcnn, cnnred, rewardcnnred,runningloss
    
    if os.path.isfile('loss_2000.csv'):

        with open('loss_2000.csv', newline='') as f:
            reader = csv.reader(f,quoting=csv.QUOTE_NONNUMERIC)
            runningloss = list(reader)[0]

        print("=> loading checkpoint... ")
       

    else:
        print("no checkpoint found...")

def loadbrain1():
    global cnn, optimizer, rewardcnn, cnnred, rewardcnnred
    if os.path.isfile('bestrandom.pth'):
        print("=> loading checkpoint... ")
        checkpoint = torch.load('bestrandom.pth', map_location=cuda0)
        cnn.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

      
        
       

        print("done !")
    else:
        print("no checkpoint found...")










