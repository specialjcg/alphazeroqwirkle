import random

import numpy as np


class BagNumpy:

    def __init__(self):
        self.tilecolor = np.zeros(108, dtype=np.int32)
        self.tileshape =np.zeros( 108, dtype=np.int32)
        j=0

        for i in range(1,4):
            for color in range(1,7):
                for shape in range(1,7):
                    self.tilecolor[j]=color
                    self.tileshape[j]=shape
                    j+=1

    def getTile(self,index):
        return self.tilecolor[index],self.tileshape[index]
    def bagCount(self):
        return len(np.where(self.tilecolor != 0)[0])
    def isEmpty(self):
        return self.bagCount() == 108
    def getRamdomTile(self):
        test=0
        while test==0:
            randomIndex = random.randrange(0, 108)
            tileRandom=self.tilecolor[randomIndex],self.tileshape[randomIndex]
            test=self.tilecolor[randomIndex]
        self.tilecolor[randomIndex] = 0
        self.tileshape[randomIndex] = 0
        return  tileRandom

    def copy(self):
        bagcopy=BagNumpy()
        bagcopy.tilecolor=np.copy(self.tilecolor)
        bagcopy.tileshape=np.copy(self.tileshape)
        return bagcopy

    def returnTileToBag(self, colorRack, shape):
        self.tilecolor[np.where(self.tilecolor==0)[0][0]]=colorRack
        self.tileshape[np.where(self.tileshape==0)[0][0]]=shape
