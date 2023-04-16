import numpy as np
import torch

from qwirckleAlphazero import memory, BATCH_SIZE, cnn, alphaloss


def train_one_epoch(epoch_index, tb_writer, optimizer):
    batch_idx = 0
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
    for i in range(len(boards)):
        # Every data instance is an input + label pair
        # boards, pis, vs = data

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
