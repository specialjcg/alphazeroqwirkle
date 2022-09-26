import numpy
import numpy as np

from BagBinaire import BagBinaire
from BagNumpy import BagNumpy
from TileBinaire import TileBinaire


class PlayerNumpy:

    def __init__(self):
        self.point = 0
        self.tilecolor = np.zeros(6, dtype=np.int32)
        self.tileshape = np.zeros(6, dtype=np.int32)

    def rackCount(self):
        return len(np.where(self.tilecolor != 0)[0])

    def addTileToRack(self, bag: BagNumpy):
        while self.rackCount()<6 and bag.bagCount() > 0:
           [self.tilecolor[np.where(self.tilecolor == 0)[0][0]],self.tileshape[np.where(self.tileshape == 0)[0][0]]]=bag.getRamdomTile()

    def getRack(self):
        return [self.tilecolor,self.tileshape]

    def delRack(self, color,shape):
        j=0
        for colorRack in self.tilecolor:
            if color == colorRack and self.tileshape[j]== shape:
                self.tilecolor[j]=0
                self.tileshape[j]=0
                break
            j+=1

    def newRack(self, bag:BagNumpy):
        j = 0
        for colorRack in self.tilecolor:
            bag.returnTileToBag(colorRack,self.tileshape[j])
            j+=1
        self.zero()
        self.addTileToRack(bag)

    def zero(self):
        self.tilecolor = np.zeros(6, dtype=np.int32)
        self.tileshape = np.zeros(6, dtype=np.int32)

    def copy(self):
        player = PlayerNumpy()

        player.tileshape=self.tileshape.copy()
        player.tilecolor=self.tilecolor.copy()
        player.point = self.point

        return player

    def addRack(self,color,shape):
        self.tilecolor[np.where(self.tilecolor == 0)[0][0]] = color
        self.tileshape[np.where(self.tileshape == 0)[0][0]] = shape

    def getRackList(self):
        return [[self.tilecolor[x],self.tileshape[x]]  for x in range(0,6) if self.tilecolor[x]!=0 ]