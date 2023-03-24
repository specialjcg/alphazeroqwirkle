# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# PyTorch TensorBoard support
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter




from torch.utils.tensorboard import SummaryWriter

from qwirckleAlphazero import local, loadcsv, loadbrain1, savebrain1, loadraindeque, localtrain, savegameboard, \
    loadraindeque2, train_one_epoch, savebraintrain, savebrainmultiprocess, cnn, loadraindequeTrain, \
    train_one_epochLoader
from ConnectNet import ConnectNet


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # loadraindeque()
    # loadbrain1()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('/home/jcgouleau/PycharmProjects/alphazeroqwirkle/fashion_trainer_{}'.format(timestamp))
    loadraindeque()
    epoch_number = 0

    EPOCHS = 5
    cnn = ConnectNet()
    cnn.init_weights()
    # optimizer = torch.optim.SGD(cnn.parameters(),lr=0.1,momentum=0.9,weight_decay=5e-4)
    # optimizer = torch.optim.SGD(cnn.parameters(), lr=0.01, momentum=0.9)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.000000001)
    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch + 1))

        # Make sure gradient tracking is on, and do a pass over the data

        avg_loss = train_one_epochLoader(epoch, writer,optimizer)
    savebrainmultiprocess(cnn,optimizer)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
