import math

import numpy as np
import torch

from isFinish import isFinish


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
        mask = action_probs > 0.0001
        actions = mask.nonzero(as_tuple=True)[0]

        # import time
        #
        # # Code before optimization
        # start_time = time.time()
        #
        # for a in actions:
        #     prob = action_probs[a]
        #     self.children[a] = Node(prior=prob.item(), to_play=self.to_play * -1,
        #                             action=indiceStateChildren[a].item())
        #
        # end_time = time.time()
        # print(f"Time taken (before optimization): {end_time - start_time} seconds")
        #
        # # Code after optimization
        # start_time = time.time()

        probs = action_probs[actions]
        indices = indiceStateChildren[actions].tolist()
        to_play = self.to_play * -1

        self.children.update({
            a.item(): Node(prior=prob.item(), to_play=to_play, action=indices[i])
            for i, (a, prob) in enumerate(zip(actions, probs))
        })

        # end_time = time.time()
        # print(f"Time taken (after optimization): {end_time - start_time} seconds")

        # for a in actions:
        #     prob = action_probs[a]
        #     self.children[a] = Node(prior=prob.item(), to_play=self.to_play * -1,
        #                             action=indiceStateChildren[a].item())

        # for a, prob in enumerate(action_probs):
        #     if prob > 0.0001:
        #         self.children[a] = Node(prior=prob.item(), to_play=self.to_play * -1,
        #                                 action=indiceStateChildren[a].item())

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
