import concurrent
import itertools

import numpy as np

from BagNumpy import BagNumpy
from PlayerNumpy import PlayerNumpy
from TileColor import TileColor
from TileShape import TileShape
from collections import defaultdict

class GameNumpy:
    def __init__(self):
        self.round = 0
        self.bag = BagNumpy()
        self.player1 = PlayerNumpy()
        self.player2 = PlayerNumpy()
        self.tilecolor = np.zeros(shape=(108, 108), dtype=np.int32)
        self.tileshape = np.zeros(shape=(108, 108), dtype=np.int32)
        self.tilecolortempory = np.zeros(shape=(108, 108), dtype=np.int32)
        self.tileshapetempory = np.zeros(shape=(108, 108), dtype=np.int32)
        self.player1.addTileToRack(self.bag)
        self.player2.addTileToRack(self.bag)
        self.isvalid = True
        self.listValidMoves = []

    def place(self, color, shape, posx, posy):
        x = posx + 54
        y = posy + 54
        if self.tilecolor[x, y] == 0:
            sameshape = True
            samecolor = True
            pos = 1
            while self.tilecolor[x, y + pos] != 0 and y+pos<108:
                if self.tilecolor[x, y + pos] == color and self.tileshape[x, y + pos] == shape:
                    return False
                samecolor = self.tilecolor[x, y + pos] == color and samecolor
                sameshape = self.tileshape[x, y + pos] == shape and sameshape
                pos += 1
            if samecolor == False and sameshape == False:
                return False
            sameshape = True
            samecolor = True
            pos = 1
            while self.tilecolor[x, y - pos] != 0 and y-pos>0:
                if self.tilecolor[x, y - pos] == color and self.tileshape[x, y - pos] == shape:
                    return False
                samecolor = self.tilecolor[x, y - pos] == color and samecolor
                sameshape = self.tileshape[x, y - pos] == shape and sameshape
                pos += 1
            pos = 1
            if samecolor == False and sameshape == False:
                return False
            sameshape = True
            samecolor = True
            while self.tilecolor[x + pos, y] != 0 and x+pos>108:
                if self.tilecolor[x + pos, y] == color and self.tileshape[x + pos, y] == shape:
                    return False
                samecolor = self.tilecolor[x + pos, y] == color and samecolor
                sameshape = self.tileshape[x + pos, y] == shape and sameshape
                pos += 1
            pos = 1
            if samecolor == False and sameshape == False:
                return False
            sameshape = True
            samecolor = True
            while self.tilecolor[x - pos, y] != 0 and x-pos>0:
                if self.tilecolor[x - pos, y] == color and self.tileshape[x - pos, y] == shape:
                    return False
                samecolor = self.tilecolor[x - pos, y] == color and samecolor
                sameshape = self.tileshape[x - pos, y] == shape and sameshape
                pos += 1
            if samecolor == False and sameshape == False:
                return False

            self.tilecolor[x, y] = color
            self.tileshape[x, y] = shape
            return True

        return False



    # def placetempory(self, color, shape, posx, posy):
    #     x = posx + 54
    #     y = posy + 54
    #     if x < 107 and y < 107:
    #         if self.tilecolortempory[x, y] == 0:
    #
    #             sameshape = True
    #             samecolor = True
    #             pos = 1
    #             while self.tilecolortempory[x, y + pos] != 0 and y + pos < 107:
    #                 if self.tilecolortempory[x, y + pos] == color and self.tileshapetempory[x, y + pos] == shape:
    #                     return False
    #                 samecolor = self.tilecolortempory[x, y + pos] == color and samecolor
    #                 sameshape = self.tileshapetempory[x, y + pos] == shape and sameshape
    #                 pos += 1
    #             if samecolor == False and sameshape == False:
    #                 return False
    #             sameshape = True
    #             samecolor = True
    #             pos = 1
    #             while self.tilecolortempory[x, y - pos] != 0 and y - pos > 0:
    #                 if self.tilecolortempory[x, y - pos] == color and self.tileshapetempory[x, y - pos] == shape:
    #                     return False
    #                 samecolor = self.tilecolortempory[x, y - pos] == color and samecolor
    #                 sameshape = self.tileshapetempory[x, y - pos] == shape and sameshape
    #                 pos += 1
    #             pos = 1
    #             if samecolor == False and sameshape == False:
    #                 return False
    #             sameshape = True
    #             samecolor = True
    #             while self.tilecolortempory[x + pos, y] != 0 and x + pos < 107:
    #                 if self.tilecolortempory[x + pos, y] == color and self.tileshapetempory[x + pos, y] == shape:
    #                     return False
    #                 samecolor = self.tilecolortempory[x + pos, y] == color and samecolor
    #                 sameshape = self.tileshapetempory[x + pos, y] == shape and sameshape
    #                 pos += 1
    #             pos = 1
    #             if samecolor == False and sameshape == False:
    #                 return False
    #             sameshape = True
    #             samecolor = True
    #             while self.tilecolortempory[x - pos, y] != 0 and x - pos > 0:
    #                 if self.tilecolortempory[x - pos, y] == color and self.tileshapetempory[x - pos, y] == shape:
    #                     return False
    #                 samecolor = self.tilecolortempory[x - pos, y] == color and samecolor
    #                 sameshape = self.tileshapetempory[x - pos, y] == shape and sameshape
    #                 pos += 1
    #             if samecolor == False and sameshape == False:
    #                 return False
    #
    #             self.tilecolortempory[x, y] = color
    #             self.tileshapetempory[x, y] = shape
    #             return True
    #
    #         return False
    #     return False



    def placetempory(self, color, shape, posx, posy):
        x = posx + 54
        y = posy + 54
        if x < 0 or x >= 107 or y < 0 or y >= 107:
            return False

        # Return False if the position is already occupied
        if self.tilecolortempory[x, y] != 0:
            return False

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        for dx, dy in directions:
            sameshape = True
            samecolor = True
            pos = 1
            while self.tilecolortempory[
                x + dx * pos, y + dy * pos] != 0 and 0 < x + dx * pos < 107 and 0 < y + dy * pos < 107:
                color_match = self.tilecolortempory[x + dx * pos, y + dy * pos] == color
                shape_match = self.tileshapetempory[x + dx * pos, y + dy * pos] == shape

                if color_match and shape_match:
                    return False

                samecolor = samecolor and color_match
                sameshape = sameshape and shape_match
                pos += 1

            if not samecolor and not sameshape:
                return False

        self.tilecolortempory[x, y] = color
        self.tileshapetempory[x, y] = shape
        return True


    # def placetempory(self, color, shape, posx, posy):
    #     x = posx + 54
    #     y = posy + 54
    #
    #     # Return False if the position is outside the bounds of the board
    #     if x < 0 or x >= 107 or y < 0 or y >= 107:
    #         return False
    #
    #     # Return False if the position is already occupied
    #     if self.tilecolortempory[x, y] != 0:
    #         return False
    #
    #     # Check if the game piece has the same color or shape as a game piece in the same row or column
    #     for i in range(107):
    #         if (self.tilecolortempory[x, i] == color or self.tileshapetempory[x, i] == shape) and (i != y):
    #             return False
    #         if (self.tilecolortempory[i, y] == color or self.tileshapetempory[i, y] == shape) and (i != x):
    #             return False
    #
    #     # Place the game piece
    #     self.tilecolortempory[x, y] = color
    #     self.tileshapetempory[x, y] = shape
    #     return True

    def listValidMovePlayer2(self):
        self.listValidMoves = []
        inp_list = self.player2.getRackList()
        permutations = []
        for i in range(1, len(inp_list) + 1):
            permutations.extend(list(itertools.permutations(inp_list, r=i)))

        if (len(permutations) > 1):
            permutations = np.unique(permutations)
            vfunc = np.vectorize(self.validTilePerùutation)

            permutations = permutations[np.where(vfunc(permutations) == True)]

        if len(np.where(self.tilecolor != 0)[0]) > 0:
            listNotZero = np.where(self.tilecolor != 0)
            val = len(listNotZero[0])

            [[self.permutationFromPositionTiletileleft(permut, listNotZero[0][x] - 54,
                                                       listNotZero[1][x] - 54) for x in range(val)] for permut in
             permutations]
        else:
            i = 0
            while i < len(permutations):
                self.deepBoardbinarryCopy()

                isvalidTempory = True
                for index, tile in enumerate(permutations[i]):
                    isvalidTempory = isvalidTempory and self.placetempory(tile[0], tile[1], 0, index)

                if isvalidTempory:
                    rackValidMove = [[tile[0], tile[1], 0, index] for index, tile in
                                     enumerate(permutations[i])]
                    self.listValidMoves.append(rackValidMove)
                    self.listValidMoves = self.unique(self.listValidMoves)
                i += 1
        self.listValidMoves=sorted(self.listValidMoves, key=lambda x: -len(x))[:10]
    def listValidMovePlayer2All(self):
        self.listValidMoves = []
        inp_list = self.player2.getRackList()
        # permutations = []
        # for i in range(1, len(inp_list) + 1):
        #     permutations.extend(list(itertools.permutations(inp_list, r=i)))
        permutations = [perm for i in range(1, len(inp_list) + 1) for perm in itertools.permutations(inp_list, r=i)]
        if (len(permutations) > 1):
            permutations = np.unique(permutations)
            vfunc = np.vectorize(self.validTilePerùutation)

            permutations = permutations[np.where(vfunc(permutations) == True)]

        if len(np.where(self.tilecolor != 0)[0]) > 0:
            listNotZero = np.where(self.tilecolor != 0)
            val = len(listNotZero[0])


            [[self.permutationFromPositionTiletileleftAll(permut, listNotZero[0][x] - 54,
                                                       listNotZero[1][x] - 54) for x in range(val)] for permut in
             permutations]
        else:
            i = 0
            while i < len(permutations):
                self.deepBoardbinarryCopy()

                isvalidTempory = True
                for index, tile in enumerate(permutations[i]):
                    isvalidTempory = isvalidTempory and self.placetempory(tile[0], tile[1], 0, index)

                if isvalidTempory:
                    rackValidMove = [[tile[0], tile[1], 0, index] for index, tile in
                                     enumerate(permutations[i])]
                    self.listValidMoves.append(rackValidMove)
                    self.listValidMoves = self.unique(self.listValidMoves)
                i += 1
        self.listValidMoves=sorted(self.listValidMoves, key=lambda x: -len(x))
    def testplaceTile(self, permutation, posx, posy, sensx, sensy):
        self.deepBoardbinarryCopy()
        for index, tile in enumerate(permutation):

            isvalidTempory = self.placetempory(tile[0], tile[1], posx + sensx * index, posy + sensy * index)
            if isvalidTempory == False:
                break
        return isvalidTempory

    def testplaceTilevoid(self, permutation, posx, posy, sensx, sensy):
        self.deepBoardnumpyVoid()
        for index, tile in enumerate(permutation):

            isvalidTempory = self.placetempory(tile[0], tile[1], posx + sensx * index, posy + sensy * index)
            if isvalidTempory == False:
                break
        return isvalidTempory

    def permute(self,indices, start, end, permutations):
        if start == end:
            permutations.append(indices.copy())
        else:
            for i in range(start, end):
                indices[start], indices[i] = indices[i], indices[start]
                self.permute(indices, start + 1, end, permutations)
                indices[start], indices[i] = indices[i], indices[start]

    def permutationFromPositionTiletileleft(self, permutation, posxtile, posytile):
        permutations = []
        indices = list(range(len(permutation)))
        self.permute(indices, 0, len(indices), permutations)

        for indices in permutations:
            # Create a list of positions surrounding the current position on the board
            positions = [
                (posxtile + 1, posytile),
                (posxtile - 1, posytile),
                (posxtile, posytile - 1),
                (posxtile, posytile + 1)
            ]

            # Loop through each position
            for posx, posy in positions:
                # Skip this position if the tile is not empty
                if self.tilecolor[posx, posy] != 0:
                    continue

                # Loop through each orientation
                for indx in range(4):
                    sensx, sensy = [(1, 0), (-1, 0), (0, -1), (0, 1)][indx]

                    # Get the permuted tiles in the current orientation
                    permuted_permutation = [permutation[i] for i in indices]

                    # Check if the permuted tiles can be placed at this position and orientation
                    if self.testplaceTile(permuted_permutation, posx, posy, sensx, sensy):
                        # Create a list of valid moves for this placement
                        rackValidMove = [
                            [tile[0], tile[1], posx + sensx * index, posy + sensy * index]
                            for index, tile in enumerate(permuted_permutation)
                        ]

                        # Add the valid moves to the list if they are not already present
                        if rackValidMove not in self.listValidMoves:
                            self.listValidMoves.append(rackValidMove)

        self.listValidMoves = sorted(self.listValidMoves, key=lambda x: -len(x))[:10]
    def permutationFromPositionTiletileleftAll(self, permutation, posxtile, posytile):
        permutations = []
        indices = list(range(len(permutation)))
        self.permute(indices, 0, len(indices), permutations)

        for indices in permutations:
            # Create a list of positions surrounding the current position on the board
            positions = [
                (posxtile + 1, posytile),
                (posxtile - 1, posytile),
                (posxtile, posytile - 1),
                (posxtile, posytile + 1)
            ]

            # Loop through each position
            for posx, posy in positions:
                # Skip this position if the tile is not empty
                if self.tilecolor[posx, posy] != 0:
                    continue

                # Loop through each orientation
                for indx in range(4):
                    sensx, sensy = [(1, 0), (-1, 0), (0, -1), (0, 1)][indx]

                    # Get the permuted tiles in the current orientation
                    permuted_permutation = [permutation[i] for i in indices]

                    # Check if the permuted tiles can be placed at this position and orientation
                    if self.testplaceTile(permuted_permutation, posx, posy, sensx, sensy):
                        # Create a list of valid moves for this placement
                        rackValidMove = [
                            [tile[0], tile[1], posx + sensx * index, posy + sensy * index]
                            for index, tile in enumerate(permuted_permutation)
                        ]

                        # Add the valid moves to the list if they are not already present
                        if rackValidMove not in self.listValidMoves:
                            self.listValidMoves.append(rackValidMove)

        self.listValidMoves = sorted(self.listValidMoves, key=lambda x: -len(x))




    def listValidMovePlayer1(self):
        self.listValidMoves = []
        inp_list = self.player1.getRackList()
        # permutations = []
        # for i in range(1, len(inp_list) + 1):
        #     permutations.extend(list(itertools.permutations(inp_list, r=i)))
        permutations = [perm for i in range(1, len(inp_list) + 1) for perm in itertools.permutations(inp_list, r=i)]
        if (len(permutations) > 1):
            permutations = np.unique(permutations)
            vfunc = np.vectorize(self.validTilePerùutation)

            permutations = permutations[np.where(vfunc(permutations) == True)]

        if len(np.where(self.tilecolor != 0)[0]) > 0:
            listNotZero = np.where(self.tilecolor != 0)
            val = len(listNotZero[0])

            [[self.permutationFromPositionTiletileleft(permut, listNotZero[0][x] - 54,
                                                       listNotZero[1][x] - 54) for x in range(val)] for permut in
             permutations]
        else:
            i = 0
            while i < len(permutations):
                self.deepBoardbinarryCopy()

                isvalidTempory = True
                for index, tile in enumerate(permutations[i]):
                    isvalidTempory = isvalidTempory and self.placetempory(tile[0], tile[1], 0, index)

                if isvalidTempory:
                    rackValidMove = [[tile[0], tile[1], 0, index] for index, tile in
                                     enumerate(permutations[i])]
                    self.listValidMoves.append(rackValidMove)
                    self.listValidMoves = self.unique(self.listValidMoves)
                i += 1
        self.listValidMoves = sorted(self.listValidMoves, key=lambda x: -len(x))[:10]
    def listValidMovePlayer1All(self):
        self.listValidMoves = []
        inp_list = self.player1.getRackList()
        # permutations = []
        # for i in range(1, len(inp_list) + 1):
        #     permutations.extend(list(itertools.permutations(inp_list, r=i)))
        permutations = [perm for i in range(1, len(inp_list) + 1) for perm in itertools.permutations(inp_list, r=i)]
        if (len(permutations) > 1):
            permutations = np.unique(permutations)
            vfunc = np.vectorize(self.validTilePerùutation)

            permutations = permutations[np.where(vfunc(permutations) == True)]

        if len(np.where(self.tilecolor != 0)[0]) > 0:
            listNotZero = np.where(self.tilecolor != 0)
            val = len(listNotZero[0])

            [[self.permutationFromPositionTiletileleftAll(permut, listNotZero[0][x] - 54,
                                                       listNotZero[1][x] - 54) for x in range(val)] for permut in
             permutations]
        else:
            i = 0
            while i < len(permutations):
                self.deepBoardbinarryCopy()

                isvalidTempory = True
                for index, tile in enumerate(permutations[i]):
                    isvalidTempory = isvalidTempory and self.placetempory(tile[0], tile[1], 0, index)

                if isvalidTempory:
                    rackValidMove = [[tile[0], tile[1], 0, index] for index, tile in
                                     enumerate(permutations[i])]
                    self.listValidMoves.append(rackValidMove)
                    self.listValidMoves = self.unique(self.listValidMoves)
                i += 1
        self.listValidMoves = sorted(self.listValidMoves, key=lambda x: -len(x))
    def unique(self, list1):

        # initialize a null list
        unique_list = []

        # traverse for all elements
        for x in list1:
            # check if exists in unique_list or not
            if x not in unique_list:
                unique_list.append(x)
        # print list
        return unique_list

    def validTilePerùutation(self, tiles):

        if len(tiles) < 2:
            return True

        # testcolor=True
        # testshape=True
        # color=tileOnBoardTempory[0].color
        # shape=tileOnBoardTempory[0].shape
        # for tile in tileOnBoardTempory:
        #     if (np.all(color!=tile.color) and testcolor):
        #             testcolor=testcolor and False
        #             break
        #     if (np.all(shape!=tile.shape) and testshape):
        #             testshape=testshape and False
        #             break

        return self.testplaceTilevoid(tiles, 0, 0, 0, 1)

    def validBoard(self):
        return self.isvalid

    def setActionprob(self):
        self.actionprob = []
        for x in range(-20, 20):
            for y in range(-20, 20):
                for direction in range(0, 4):
                    if direction == 0:
                        dirx = 1
                        diry = 0
                    if direction == 1:
                        dirx = 0
                        diry = 1
                    if direction == 2:
                        dirx = -1
                        diry = 0
                    if direction == 3:
                        dirx = 0
                        diry = -1
                    for color in TileColor:
                        for j in range(0, 6):
                            tile1 = []
                            for k in range(0, j + 1):
                                tile1.append([TileColor[color], 0, x + k * dirx, y + k * diry])
                            self.actionprob.append(tile1)
                    for shape in TileShape:
                        for j in range(0, 6):
                            tile1 = []
                            for k in range(0, j + 1):
                                tile1.append([0, TileShape[shape], x + k * dirx, y + k * diry])
                            self.actionprob.append(tile1)

    def setActionprobtest(self):

        self.actionprob = []
        inp_list = range(36)
        global alltile
        alltile = [p for p in itertools.product([1, 2, 3, 4, 5, 6], repeat=2)]

        for i in range(1, 7):
            self.actionprob.append(itertools.combinations(inp_list, r=i))

        enumerateactionprog = self.actionprob.copy()
        self.actionprob = []

        for enum in range(0, 6):
            # val = list(enumerateactionprog[5])
            [self.goodGame(arg) for arg in enumerateactionprog[enum]]

            # [self.goodGame(alltile, dirx, diry, tiles, 0, 0) for tiles in val]

        #
        #             for tile1color in range(1,7):
        #                 for tile1shape in range(1,7):
        #                     for j in range(0,6):
        #                         tile1=[]
        #                         for k in range(0,j+1):
        #                             tile1.append([TileColor[color],TileShape[shape],x+k*dirx,y+k*diry])
        #                         self.actionprob.append(tile1)
        # enumerateactionprog=self.actionprob.copy()
        # self.actionprob = []
        # for i in range(len(enumerateactionprog)):
        #     isvalidTempory=True
        #     self.tilecolortempory = np.zeros(shape=(108, 108), dtype=np.int32)
        #     self.tileshapetempory = np.zeros(shape=(108, 108), dtype=np.int32)
        #     for index, tile in enumerate(enumerateactionprog[i]):
        #         isvalidTempory = isvalidTempory and self.placetempory(tile[0], tile[1], tile[2], tile[3])
        #
        #     if isvalidTempory:
        #         self.actionprob.append(enumerateactionprog[i])

    def goodGame(self, tiles):

        x = 0
        y = 0

        dirx = 1
        diry = 0
        tile1 = []
        isvalidTempory = True
        self.tilecolortempory = np.zeros(shape=(108, 108), dtype=np.int32)
        self.tileshapetempory = np.zeros(shape=(108, 108), dtype=np.int32)
        for k in range(0, len(np.array(tiles))):
            if isvalidTempory:
                tile1.append([alltile[np.array(tiles).item(k)][0], alltile[np.array(tiles).item(k)][1]])

                isvalidTempory = isvalidTempory and self.placetempory(alltile[np.array(tiles).item(k)][0],
                                                                      alltile[np.array(tiles).item(k)][1],
                                                                      x + k * dirx, y + k * diry)
            else:
                break
        if isvalidTempory:
            tilespermut = list(itertools.permutations(tile1))
            for tile in tilespermut:
                self.actionprob.append(tile)
            # with open('combi.csv', 'a', encoding='UTF8', newline='') as f:
            #     writer = csv.writer(f)
            #     tilespermut=list(itertools.permutations(tile1))
            #
            #     # write multiple rows
            #     for tile in tilespermut:
            #      writer.writerow(tile)
            # f.close()

    def deepBoardbinarryCopy(self):
        self.tilecolortempory = np.copy(self.tilecolor)
        self.tileshapetempory = np.copy(self.tileshape)

        pass

    def deepBoardnumpyVoid(self):
        self.tilecolortempory = np.zeros(shape=(108, 108), dtype=np.int32)
        self.tileshapetempory = np.zeros(shape=(108, 108), dtype=np.int32)

        pass

    def test3round(self):
        return self.round < 3
        pass

    def winner(self):
        if np.all(self.player1.tilecolor == 0):
            self.player1.point+=6
        if np.all(self.player2.tilecolor == 0):
            self.player2.point+=6
        if self.player2.point>self.player1.point:
            return -1
        return 1

    def __copy__(self):
        newcopy = GameNumpy()
        newcopy.bag = self.bag.copy()
        newcopy.player1 = self.player1.copy()
        newcopy.player2 = self.player2.copy()
        newcopy.tilecolor = np.copy(self.tilecolor)
        newcopy.tileshape = np.copy(self.tileshape)
        newcopy.listValidMoves = self.listValidMoves.copy()
        newcopy.isvalid = self.isvalid
        newcopy.actionprob = self.actionprob.copy()
        return newcopy

    def getpoint(self, tiles):
        point=len(tiles)
        if len(tiles) == 6:
            point +=  6
        for [x, y] in tiles:
            x1 = x + 54
            y1 = y + 54
            test = True
            test2=True
            qwirkle = 0
            for j in range(1, 7):

                if self.tilecolor[x1 - j, y1] != 0 and [x-1,y] not in tiles:
                    qwirkle+=1
                    point = point + 1
                    test2 = False
                else:
                    if (qwirkle==5):
                        point+=6
                    break
            qwirkle = 0
            for j in range(1, 7):
                if self.tilecolor[x1 + j, y1] != 0 and [x+1,y] not in tiles :
                    qwirkle += 1
                    point = point + 1
                    test2 = False
                else:
                    if (qwirkle==5):
                        point+=6
                    break
            qwirkle = 0

            for j in range(1, 7):
                if self.tilecolor[x1, y1 - j] != 0 and [x,y-1] not in tiles:
                    qwirkle += 1
                    point = point + 1
                    test2 = False
                else:
                    if (qwirkle==5):
                        point+=6
                    break
            qwirkle = 0

            for j in range(1, 7):
                if self.tilecolor[x1, y1 + j] != 0 and [x,y+1] not in tiles:
                    qwirkle += 1
                    point = point + 1
                    test2 = False
                else:
                    if (qwirkle==5):
                        point+=6
                    break

        return point
