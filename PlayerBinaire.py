import numpy

from BagBinaire import BagBinaire
from TileBinaire import TileBinaire


class PlayerBinaire:

    def __init__(self):
        self.point = 0
        self.rack = []

    def addTileToRack(self, bag: BagBinaire):
        while (len(self.rack) <= 5 and len(bag.bag) > 0):
            self.rack.append(bag.getRamdomTile())

    def getRack(self):
        return [[i.color,i.shape] for i in self.rack]

    def delRack(self, tile: TileBinaire):
        for rackdel in self.rack:
            if rackdel.shape == tile.shape and rackdel.color == tile.color:
                self.rack.remove(rackdel)
                break

    def newRack(self, bag):
        for rack in self.rack:
            bag.bag.append(rack)
        self.rack = []
        self.addTileToRack(bag)

    def copy(self):
        player = PlayerBinaire()
        player.rack = []
        for l in self.getRack():
            player.rack.append(l)
        player.point = self.point

        return player