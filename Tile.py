from Bag import TileColor
from Coordinate import Coordinate
from Bag import TileShape


class Tile:
    def __init__(self, color: TileColor, shape: TileShape, coordinate: Coordinate):
        self.color = color
        self.shape = shape
        self.coordinate = coordinate

    def get(self):
        return [self.color, self.shape, [self.coordinate.x, self.coordinate.y]]