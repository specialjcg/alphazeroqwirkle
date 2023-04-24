# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from qwirckleAlphazero import local, loadcsv, loadbrain1, savebrain1, savebraindequeZero, savebraindeque, loadraindeque
from loadbrain2 import loadbrain2

from graphviz import Digraph
class Node:
    def __init__(self, value=None):
        self.value = value
        self.children = []

class Tree:
    def __init__(self, root=None):
        self.root = root

    def add_node(self, value, parent_value):
        new_node = Node(value)
        parent_node = self.find_node(parent_value)
        parent_node.children.append(new_node)

    def find_node(self, value):
        queue = [self.root]
        while queue:
            current_node = queue.pop(0)
            if current_node.value == value:
                return current_node
            queue += current_node.children

# create the tree


def visualize_tree(root):
    dot = Digraph()
    queue = [(root, None)]
    while queue:
        current_node, parent_node = queue.pop(0)
        dot.node(current_node.value)
        if parent_node:
            dot.edge(parent_node.value, current_node.value)
        queue += [(child, current_node) for child in current_node.children]
    return dot



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    root = Node("A")
    tree = Tree(root)

    # add nodes to the tree
    tree.add_node("B", "A")
    tree.add_node("C", "A")
    tree.add_node("D", "A")
    tree.add_node("E", "B")
    tree.add_node("F", "C")
    tree.add_node("G", "C")
    dot = visualize_tree(root)
    dot.render("tree.gv", view=True)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
