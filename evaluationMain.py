# This is a sample Python script.
from qwirckleAlphazero import loadbrain1, localevaluation, getmaxWin
from loadbrain2 import loadbrain2


# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.



def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    loadbrain1()
    loadbrain2()
    localevaluation(10)


    print_hi(getmaxWin())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
