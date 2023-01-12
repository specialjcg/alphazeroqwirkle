import unittest

import numpy as np

from GameNumpy import GameNumpy


class TestPlacetempory(unittest.TestCase):

    def test_placetempory(self):
        # Set up the test environment
        game = GameNumpy()
        game.tilecolortempory = np.zeros((107, 107), dtype=int)
        game.tileshapetempory = np.zeros((107, 107), dtype=int)

        # Test placing a tile outside the board
        self.assertFalse(game.placetempory(1, 1, -1, 0))
        self.assertFalse(game.placetempory(1, 1, 0, -1))
        self.assertFalse(game.placetempory(1, 1, 107, 0))
        self.assertFalse(game.placetempory(1, 1, 0, 107))

        # Test placing a tile on an already occupied position
        game.tilecolortempory[54, 54] = 1
        game.tileshapetempory[54, 54] = 1
        self.assertFalse(game.placetempory(1, 1, 0, 0))

        # Test placing a tile that violates the same shape or color rule
        self.assertTrue(game.placetempory(1, 2, 0, 0))
        self.assertFalse(game.placetempory(1, 2, 0, 1))
        self.assertFalse(game.placetempory(2, 1, 0, 1))

        # Test placing a tile that follows the rules
        self.assertTrue(game.placetempory(2, 2, 0, 1))


if __name__ == '__main__':
    unittest.main()
