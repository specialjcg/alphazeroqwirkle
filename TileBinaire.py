from Coordinate import Coordinate


class TileBinaire:
    def __init__(self, color: int, shape: int, coordinate: Coordinate = Coordinate(0, 0)):
        self.color = color
        self.shape = shape
        self.coordinate = coordinate

    def get(self):
        return self
