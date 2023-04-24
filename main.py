# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from qwirckleAlphazero import local, loadcsv, loadbrain1, savebrain1, savebraindequeZero, savebraindeque, loadraindeque
from loadbrain2 import loadbrain2


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    loadbrain1()
    loadbrain2()
    #loadraindeque()
    local(4)

    # savebrain1()
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
