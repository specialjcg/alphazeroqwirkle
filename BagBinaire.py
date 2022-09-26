from Coordinate import Coordinate
from TileBinaire import TileBinaire
import random

class BagBinaire:

    def __init__(self):
        self.bag = []

        for i in range(1,4):
            for color in range(1,7):
                for shape in range(1,7):

                    tile=TileBinaire(color, shape,Coordinate(0,0))
                    self.bag.append(tile)

    def getTile(self,index):
        return self.bag[index]
    def isEmpty(self):
        return len(self.bag) == 0
    def getRamdomTile(self):
        randomIndex=random.randrange(0, len(self.bag), 2)
        tileRandom=self.bag[randomIndex]
        del self.bag[randomIndex]
        return  tileRandom

    def copy(self):
        bagcopy=BagBinaire()
        bagcopy.bag=[]
        for bg in self.bag:
            bagcopy.bag.append(bg.get())
        return bagcopy