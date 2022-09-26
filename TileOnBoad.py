

import TileColor

import TileShape
from Coordinate import Coordinate
from TileBinaire import TileBinaire


class TileOnBoard:
    def __init__(self):
        self.tiles=[]

    def __add__(self, tile:TileBinaire):
        self.tiles.append(tile)

    def getTile(self,coordinate:Coordinate):
        for tile in self.tiles:
            if tile.coordinate.x==coordinate.x and tile.coordinate.y==coordinate.y:
                return tile
                break
        return None

    def ocupied(self,coordinate:Coordinate):
        for tile in self.tiles:
            if tile.coordinate.x == coordinate.x and tile.coordinate.y == coordinate.y:
                return True
                break
        return False

    def copy(self):
        tileboardcopy = TileOnBoard()
        tileboardcopy.tiles = []
        [tileboardcopy.__add__(tile) for tile in self.tiles]
        return tileboardcopy

    def tileEqual(self, coordinate:Coordinate,tile:TileBinaire):
        tileInCoordinate:TileBinaire=self.getTile(coordinate)
        if tileInCoordinate!=None:
            if tileInCoordinate.color==tile.color and tile.shape==tileInCoordinate.shape:
                return True
            else:
                return False

        return None

    def tileNotEqual(self, coordinate: Coordinate, tile: TileBinaire):
        tileInCoordinate: TileBinaire = self.getTile(coordinate)
        if tileInCoordinate != None:
            if tileInCoordinate.color != tile.color and tile.shape != tileInCoordinate.shape:
                return True

        return False

    def sametilecolor(self, coordinate: Coordinate, tile: TileBinaire):
        tileInCoordinate: TileBinaire = self.getTile(coordinate)
        if tileInCoordinate != None:
            if tileInCoordinate.color != tile.color:
                return False
        return True

    def sametileshape(self, coordinate: Coordinate, tile: TileBinaire):
        tileInCoordinate: TileBinaire = self.getTile(coordinate)
        if tileInCoordinate != None:
            if tileInCoordinate.shape != tile.shape:
                return False
        return True