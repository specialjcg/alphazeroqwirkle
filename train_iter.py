# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# PyTorch TensorBoard support
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import cnn_iter
from alphaloss import AlphaLoss

BATCH_SIZE = 16
def savebrainmultiprocess(cnn, optimizer):
    global savefile

    print("=> saving checkpoint... ")
    checkpoint = {'model': cnn,
                  'state_dict': cnn.state_dict(),
                  'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, 'bestrandomiter.pth')

    print("=> saving checkpoint... ")
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
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
def train_one_epochLoader(epoch_index, tb_writer, optimizer):
    pi_losses = []
    v_losses = []
    running_loss = 0.
    last_loss = 0.
    # training_loader = torch.utils.data.DataSet(memory, batch_size=4, shuffle=True, num_workers=2)
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    sample_indices = torch.randperm(len(memory))[:BATCH_SIZE]
    boards, pis, vs = list(zip(*[memory[i] for i in sample_indices]))
    statesignal = torch.empty(BATCH_SIZE, 26, 54, 54, dtype=torch.float32)
    for i, board in enumerate(boards):
        last_signal = torch.tensor(board, dtype=torch.float32)
        statesignal[i] = last_signal.reshape(26, 54, 54)

        # Every data instance is an input + label pair
        # boards, pis, vs = data

        # boards = torch.FloatTensor(np.stack(boards,axis=0))
        # boards = torch.FloatTensor(boards).cuda()

        boards_list = []
        for board in boards:
            last_signal = torch.tensor(board, dtype=torch.float32)
            boards_list.append(last_signal.reshape(26, 54, 54))
        statesignal = torch.stack(boards_list)

        pis_list = []
        for policy in pis:
            valid_moves = np.zeros(23436)
            valid_moves[policy] = 1
            policy = torch.tensor(valid_moves, dtype=torch.float32)
            pis_list.append(policy)
        target_pis = torch.stack(pis_list)

        vs_list = []
        for value in vs:
            value = torch.tensor(value, dtype=torch.float32)
            vs_list.append(value)
        target_vs = torch.stack(vs_list)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        out_pi, out_v = cnn.forward(statesignal)
        total_error = alphaloss(out_v, target_vs, out_pi, target_pis)
        total_error.backward()
        pi_losses.append(float(total_error))
        # Adjust learning weights
        optimizer.step()
        print('Policy Loss:{:.2f}'.format(np.mean(pi_losses)))

        # Gather data and report
        running_loss += total_error.item()
        if i % 32 == 31:
            last_loss = running_loss / 8  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(boards) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


if __name__ == '__main__':

    # loadraindeque()
    # loadbrain1()
    alphaloss = AlphaLoss()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('/home/jcgouleau/PycharmProjects/alphazeroqwirkle/fashion_trainer_{}'.format(timestamp))
    loadraindeque()
    epoch_number = 0

    EPOCHS = 5
    cnn = cnn_iter.ConnectNet()
    cnn.init_weights()
    # optimizer = torch.optim.SGD(cnn.parameters(),lr=0.1,momentum=0.9,weight_decay=5e-4)
    # optimizer = torch.optim.SGD(cnn.parameters(), lr=0.01, momentum=0.9)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.000000001)
    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch + 1))

        # Make sure gradient tracking is on, and do a pass over the data

        avg_loss = train_one_epochLoader(epoch, writer, optimizer)
    savebrainmultiprocess(cnn, optimizer)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
