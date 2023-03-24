# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import datetime
from multiprocessing import Process

from qwirckleAlphazero import local, loadbrain1, loadbrain2
import multiprocessing

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    start_time = datetime.datetime.now().time().strftime('%H:%M:%S')
    with multiprocessing.Pool(processes=8) as pool:
        loadbrain1()
        loadbrain2()
        # loadraindeque()
        pool.map(local, [5,5,5,5,5,5,5,5])
        pool.close()  # no more tasks
        pool.join()  # wrap up current tasks
    end_timeTotal = datetime.datetime.now().time().strftime('%H:%M:%S')
    total_timemulti = (datetime.datetime.strptime(end_timeTotal, '%H:%M:%S') - datetime.datetime.strptime(start_time,
                                                                                                '%H:%M:%S'))
    print('total_time:' + str(total_timemulti))
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
