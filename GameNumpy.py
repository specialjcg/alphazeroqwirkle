import datetime
import itertools
import timeit
import csv
from numpy import int8

from Bag import deepBoardCopy
from BagBinaire import BagBinaire
from BagNumpy import BagNumpy
from Coordinate import Coordinate
from PlayerBinaire import PlayerBinaire
import numpy as np
import copy

from PlayerNumpy import PlayerNumpy
from Tile import Tile
from TileBinaire import TileBinaire
from TileColor import TileColor
from TileOnBoad import TileOnBoard
from TileShape import TileShape


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
            while self.tilecolor[x, y + pos] != 0:
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
            while self.tilecolor[x, y - pos] != 0:
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
            while self.tilecolor[x + pos, y] != 0:
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
            while self.tilecolor[x - pos, y] != 0:
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

    def placetempory(self, color, shape, posx, posy):
        x = posx + 54
        y = posy + 54
        if x < 107 and y < 107:
            if self.tilecolortempory[x, y] == 0:

                sameshape = True
                samecolor = True
                pos = 1
                while self.tilecolortempory[x, y + pos] != 0 and y + pos < 107:
                    if self.tilecolortempory[x, y + pos] == color and self.tileshapetempory[x, y + pos] == shape:
                        return False
                    samecolor = self.tilecolortempory[x, y + pos] == color and samecolor
                    sameshape = self.tileshapetempory[x, y + pos] == shape and sameshape
                    pos += 1
                if samecolor == False and sameshape == False:
                    return False
                sameshape = True
                samecolor = True
                pos = 1
                while self.tilecolortempory[x, y - pos] != 0 and y - pos > 0:
                    if self.tilecolortempory[x, y - pos] == color and self.tileshapetempory[x, y - pos] == shape:
                        return False
                    samecolor = self.tilecolortempory[x, y - pos] == color and samecolor
                    sameshape = self.tileshapetempory[x, y - pos] == shape and sameshape
                    pos += 1
                pos = 1
                if samecolor == False and sameshape == False:
                    return False
                sameshape = True
                samecolor = True
                while self.tilecolortempory[x + pos, y] != 0 and x + pos < 107:
                    if self.tilecolortempory[x + pos, y] == color and self.tileshapetempory[x + pos, y] == shape:
                        return False
                    samecolor = self.tilecolortempory[x + pos, y] == color and samecolor
                    sameshape = self.tileshapetempory[x + pos, y] == shape and sameshape
                    pos += 1
                pos = 1
                if samecolor == False and sameshape == False:
                    return False
                sameshape = True
                samecolor = True
                while self.tilecolortempory[x - pos, y] != 0 and x - pos > 0:
                    if self.tilecolortempory[x - pos, y] == color and self.tileshapetempory[x - pos, y] == shape:
                        return False
                    samecolor = self.tilecolortempory[x - pos, y] == color and samecolor
                    sameshape = self.tileshapetempory[x - pos, y] == shape and sameshape
                    pos += 1
                if samecolor == False and sameshape == False:
                    return False

                self.tilecolortempory[x, y] = color
                self.tileshapetempory[x, y] = shape
                return True

            return False

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

    def permutationFromPositionTiletileleft(self, permutation, posxtile, posytile):
        for j in range(0, 4):
            if j == 0:
                posx = posxtile + 1
                posy = posytile
            if j == 1:
                posx = posxtile - 1
                posy = posytile
            if j == 2:
                posx = posxtile
                posy = posytile - 1
            if j == 3:
                posx = posxtile
                posy = posytile + 1

            if self.tilecolor[posx, posy] == 0:
                for indx in range(0, 4):
                    if indx == 0:
                        sensx = -1
                        sensy = 0
                    if indx == 1:
                        sensx = 1
                        sensy = 0
                    if indx == 2:
                        sensx = 0
                        sensy = -1
                    if indx == 3:
                        sensx = 0
                        sensy = 1

                    if self.testplaceTile(permutation, posx, posy, sensx, sensy):
                        rackValidMove = [[tile[0], tile[1], posx + sensx * index, posy + sensy * index] for index, tile
                                         in
                                         enumerate(permutation)]
                        self.listValidMoves.append(rackValidMove)
                        self.listValidMoves = self.unique(self.listValidMoves)

    def listValidMovePlayer1(self):
        self.listValidMoves = []
        inp_list = self.player1.getRackList()
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
            for j in range(1, 7):
                if self.tilecolor[x1 - j, y1] != 0 and [x-1,y] not in tiles:
                    point = point + 1
                    test2 = False
                else:
                    break

            for j in range(1, 7):
                if self.tilecolor[x1 + j, y1] != 0 and [x+1,y] not in tiles :
                    point = point + 1
                    test2 = False
                else:
                    break

            for j in range(1, 7):
                if self.tilecolor[x1, y1 - j] != 0 and [x,y-1] not in tiles:
                    point = point + 1
                    test2 = False
                else:
                    break

            for j in range(1, 7):
                if self.tilecolor[x1, y1 + j] != 0 and [x,y+1] not in tiles:
                    point = point + 1
                    test2 = False
                else:
                    break

        return point
