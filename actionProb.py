# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import GameNumpy as newGame
import pickle
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    game=newGame.GameNumpy()
    game.setActionprobtest()




    pickle.dump(game.actionprob, open('gameActionProb.pkl', 'wb'))
    print("=> saving gameActionProb... ")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
