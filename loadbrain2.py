import os

import torch

from cnn_iter import ConnectNet

cnn_iter1 = ConnectNet()
cnn_iter1.init_weights()
def loadbrain2():
    global cnn_iter1, optimizeriter

    optimizeriter = torch.optim.Adam(cnn_iter1.parameters(), lr=0.000001)
    if os.path.isfile('bestrandomiter.pth'):
        print("=> loading checkpoint... ")
        # checkpoint = torch.load('bestrandom.pth', map_location=cuda0)
        checkpoint = torch.load('bestrandomiter.pth')
        cnn_iter1.load_state_dict(checkpoint['state_dict'])
        optimizeriter.load_state_dict(checkpoint['optimizer'])

        print("done !")
    else:
        print("no checkpoint found...")
