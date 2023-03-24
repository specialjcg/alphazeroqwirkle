import datetime
import itertools
import timeit

from numpy import int8

from Bag import deepBoardCopy
from BagBinaire import BagBinaire
from Coordinate import Coordinate
from PlayerBinaire import PlayerBinaire
import numpy as np
import copy
from Tile import Tile
from TileBinaire import TileBinaire
from TileColor import TileColor
from TileOnBoad import TileOnBoard
from TileShape import TileShape


class GameBinaire:
    def __init__(self):
        self.boardTempory = []
        self.board = []
        self.bag = BagBinaire()
        self.player1 = PlayerBinaire()
        self.player2 = PlayerBinaire()
        self.tileOnBoard=TileOnBoard()
        self.tileOnBoardTempory=[]
        self.listValidMoves=[]
        self.player1.addTileToRack(self.bag)
        self.player2.addTileToRack(self.bag)
        self.isvalid =True
        self.isvalidTempory =True
        self.actionprob=[]

    def place(self,tile: TileBinaire,x,y):
        placex=x
        placey=y

        if (self.tileOnBoard.ocupied(Coordinate(placex, placey))==False):
            tileup = self.tileOnBoard.getTile(Coordinate(placex, placey - 1))
            tiledown = self.tileOnBoard.getTile(Coordinate(placex, placey + 1))
            tileleft = self.tileOnBoard.getTile(Coordinate(placex - 1, placey))
            tileright = self.tileOnBoard.getTile(Coordinate(placex + 1, placey))
            tileSE = self.tileOnBoard.getTile(Coordinate(placex + 1, placey + 1))
            tileSO = self.tileOnBoard.getTile(Coordinate(placex - 1, placey + 1))
            tileNE = self.tileOnBoard.getTile(Coordinate(placex + 1, placey - 1))
            tileNO = self.tileOnBoard.getTile(Coordinate(placex - 1, placey - 1))
            if tileup != None:
                if (
                        tile.color != 0 and tileup.color != 0 and tileup.color != tile.color and tileup.shape != tile.shape):
                    return False
            if tiledown != None:
                if (
                        tile.color != 0 and tiledown.color != 0 and tiledown.color != tile.color and tiledown.shape != tile.shape):
                    return False
            if tileright != None:
                if (
                        tile.color != 0 and tileright.color != 0 and tileright.color != tile.color and tileright.shape != tile.shape):
                    return False
            if tileleft != None:
                if (
                        tile.color != 0 and tileleft.color != 0 and tileleft.color != tile.color and tileleft.shape != tile.shape):
                    return False
            if tileright == None and tiledown == None:
                if (tile.color != 0 and tileSE != None):
                    return False
            if tileright == None and tileup == None:
                if (tile.color != 0 and tileNE != None):
                    return False
            if tileleft == None and tiledown == None:
                if (tile.color != 0 and tileSO != None):
                    return False
            if tileleft == None and tileup == None:
                if (tile.color != 0 and tileNO != None):
                    return False

            if (tileup != None and tiledown != None):
                if ((tile.color == tileup.color and tile.color == tiledown.color) and (
                        tile.shape == tileup.shape and tile.shape == tiledown.shape)):
                    return False
            if (tileright != None and tileleft != None):
                if ((tile.color == tileright.color and tile.color == tileleft.color) and (
                        tile.shape == tileright.shape and tile.shape == tileleft.shape)):
                    return False

            if (tileup == None and tiledown == None and tileleft == None and tileright == None and len(
                    self.tileOnBoard.tiles) > 0):
                return False
            sameshape=True
            samecolor=True
            for pos in range(1,7):

                if self.tileOnBoard.tileEqual(Coordinate(placex, placey-pos),tile):
                    return  False
                if self.tileOnBoard.tileNotEqual(Coordinate(placex, placey - pos), tile):
                    return False
                samecolor = self.tileOnBoard.sametilecolor(Coordinate(placex, placey - pos), tile) and samecolor
                sameshape = self.tileOnBoard.sametileshape(Coordinate(placex, placey - pos), tile) and sameshape
            if  samecolor==False and sameshape==False and len(self.tileOnBoard.tiles)>0:
                return False
            sameshape = True
            samecolor = True
            for pos in range(1, 7):

                if self.tileOnBoard.tileEqual(Coordinate(placex, placey + pos), tile):
                    return False
                if self.tileOnBoard.tileNotEqual(Coordinate(placex, placey + pos), tile):
                    return False
                samecolor = self.tileOnBoard.sametilecolor(Coordinate(placex, placey + pos), tile) and samecolor
                sameshape = self.tileOnBoard.sametileshape(Coordinate(placex, placey + pos), tile) and sameshape
            if  samecolor == False and sameshape == False and len(self.tileOnBoard.tiles)>0:
                return False
            sameshape = True
            samecolor = True
            for pos in range(1, 7):

                if self.tileOnBoard.tileEqual(Coordinate(placex+pos, placey ), tile):
                    return False
                if self.tileOnBoard.tileNotEqual(Coordinate(placex+pos, placey) , tile):
                    return False
                samecolor = self.tileOnBoard.sametilecolor(Coordinate(placex+pos, placey), tile) and samecolor
                sameshape = self.tileOnBoard.sametileshape(Coordinate(placex+pos, placey), tile) and sameshape
            if samecolor == False and sameshape == False and len(self.tileOnBoard.tiles)>0:
                return False
            sameshape = True
            samecolor = True
            for pos in range(1, 7):
                if self.tileOnBoard.getTile(Coordinate(placex-pos, placey ))==None:
                    break
                if self.tileOnBoard.tileEqual(Coordinate(placex-pos, placey), tile):
                    return False
                if self.tileOnBoard.tileNotEqual(Coordinate(placex-pos, placey), tile):
                    return False
                samecolor = self.tileOnBoard.sametilecolor(Coordinate(placex-pos, placey), tile) and samecolor
                sameshape = self.tileOnBoard.sametileshape(Coordinate(placex-pos, placey), tile) and sameshape
            if samecolor == False and sameshape == False and len(self.tileOnBoard.tiles)>0:
                return False




        else:
            return False


        self.tileOnBoard.__add__(TileBinaire(tile.color,tile.shape,Coordinate(placex, placey)))
        self.board.append(Tile(list(TileColor.keys())[list(TileColor.values()).index(tile.color)]
                               , list(TileShape.keys())[list(TileShape.values()).index(tile.shape)], Coordinate(x,y)))
        return True

    def placetempory(self,tile: TileBinaire,x,y):
        placex = x
        placey = y

        if (self.tileOnBoardTempory.ocupied(Coordinate(placex, placey)) == False):
            tileup = self.tileOnBoardTempory.getTile(Coordinate(placex, placey - 1))
            tiledown = self.tileOnBoardTempory.getTile(Coordinate(placex, placey + 1))
            tileleft = self.tileOnBoardTempory.getTile(Coordinate(placex - 1, placey))
            tileright = self.tileOnBoardTempory.getTile(Coordinate(placex + 1, placey))
            tileSE = self.tileOnBoardTempory.getTile(Coordinate(placex + 1, placey + 1))
            tileSO = self.tileOnBoardTempory.getTile(Coordinate(placex - 1, placey + 1))
            tileNE = self.tileOnBoardTempory.getTile(Coordinate(placex + 1, placey - 1))
            tileNO = self.tileOnBoardTempory.getTile(Coordinate(placex - 1, placey - 1))

            if tileup != None:
                if (
                        tile.color != 0 and tileup.color != 0 and tileup.color != tile.color and tileup.shape != tile.shape):
                    return False
            if tiledown != None:
                if (
                        tile.color != 0 and tiledown.color != 0 and tiledown.color != tile.color and tiledown.shape != tile.shape):
                    return False
            if tileright != None:
                if (
                        tile.color != 0 and tileright.color != 0 and tileright.color != tile.color and tileright.shape != tile.shape):
                    return False
            if tileleft != None:
                if (
                        tile.color != 0 and tileleft.color != 0 and tileleft.color != tile.color and tileleft.shape != tile.shape):
                    return False
            if tileright == None and tiledown == None:
                if (tile.color != 0 and tileSE != None):
                    return False
            if tileright == None and tileup == None:
                if (tile.color != 0 and tileNE != None):
                    return False
            if tileleft == None and tiledown == None:
                if (tile.color != 0 and tileSO != None):
                    return False
            if tileleft == None and tileup == None:
                if (tile.color != 0 and tileNO != None):
                    return False

            if (tileup != None and tiledown != None):
                if ((tile.color == tileup.color and tile.color == tiledown.color) and (
                        tile.shape == tileup.shape and tile.shape == tiledown.shape)):
                    return False
            if (tileright != None and tileleft != None):
                if ((tile.color == tileright.color and tile.color == tileleft.color) and (
                        tile.shape == tileright.shape and tile.shape == tileleft.shape)):
                    return False

            if (tileup == None and tiledown == None and tileleft == None and tileright == None and len(
                    self.tileOnBoardTempory.tiles) > 0):
                return False
            sameshape = True
            samecolor = True
            for pos in range(1, 7):
                if self.tileOnBoardTempory.getTile(Coordinate(placex, placey - pos))==None:
                    break
                if self.tileOnBoardTempory.tileEqual(Coordinate(placex, placey - pos), tile):
                    return False
                if self.tileOnBoardTempory.tileNotEqual(Coordinate(placex, placey - pos), tile):
                    return False
                samecolor = self.tileOnBoardTempory.sametilecolor(Coordinate(placex, placey - pos), tile) and samecolor
                sameshape = self.tileOnBoardTempory.sametileshape(Coordinate(placex, placey - pos), tile) and sameshape
            if samecolor == False and sameshape == False and len(self.tileOnBoardTempory.tiles) > 0:
                return False
            sameshape = True
            samecolor = True
            for pos in range(1, 7):
                if self.tileOnBoardTempory.getTile(Coordinate(placex, placey + pos))==None:
                    break
                if self.tileOnBoardTempory.tileEqual(Coordinate(placex, placey + pos), tile):
                    return False
                if self.tileOnBoardTempory.tileNotEqual(Coordinate(placex, placey + pos), tile):
                    return False
                samecolor = self.tileOnBoardTempory.sametilecolor(Coordinate(placex, placey + pos), tile) and samecolor
                sameshape = self.tileOnBoardTempory.sametileshape(Coordinate(placex, placey + pos), tile) and sameshape
            if samecolor == False and sameshape == False and len(self.tileOnBoardTempory.tiles) > 0:
                return False
            sameshape = True
            samecolor = True
            for pos in range(1, 7):
                if self.tileOnBoardTempory.getTile(Coordinate(placex +pos, placey ))==None:
                    break
                if self.tileOnBoardTempory.tileEqual(Coordinate(placex + pos, placey), tile):
                    return False
                if self.tileOnBoardTempory.tileNotEqual(Coordinate(placex + pos, placey), tile):
                    return False
                samecolor = self.tileOnBoardTempory.sametilecolor(Coordinate(placex + pos, placey), tile) and samecolor
                sameshape = self.tileOnBoardTempory.sametileshape(Coordinate(placex + pos, placey), tile) and sameshape
            if samecolor == False and sameshape == False and len(self.tileOnBoardTempory.tiles) > 0:
                return False
            sameshape = True
            samecolor = True
            for pos in range(1, 7):
                if self.tileOnBoardTempory.getTile(Coordinate(placex-pos , placey))==None:
                    break
                if self.tileOnBoardTempory.tileEqual(Coordinate(placex - pos, placey), tile):
                    return False
                if self.tileOnBoardTempory.tileNotEqual(Coordinate(placex - pos, placey), tile):
                    return False
                samecolor = self.tileOnBoardTempory.sametilecolor(Coordinate(placex - pos, placey), tile) and samecolor
                sameshape = self.tileOnBoardTempory.sametileshape(Coordinate(placex - pos, placey), tile) and sameshape
            if samecolor == False and sameshape == False and len(self.tileOnBoardTempory.tiles) > 0:
                return False




        else:
            return False

        self.tileOnBoardTempory.__add__(TileBinaire(tile.color, tile.shape, Coordinate(placex, placey)))
        # self.board.append(Tile(list(TileColor.keys())[list(TileColor.values()).index(tile.color)]
        #                        , list(TileShape.keys())[list(TileShape.values()).index(tile.shape)], Coordinate(x, y)))
        return True


    def __copy__(self):
        newcopy = GameBinaire()
        newcopy.bag=self.bag.copy()
        newcopy.player1=self.player1.copy()
        newcopy.player2 = self.player2.copy()
        newcopy.tileOnBoard=self.tileOnBoard.copy()
        newcopy.tileOnBoardTempory = self.tileOnBoardTempory.copy()
        newcopy.listValidMoves = self.listValidMoves.copy()
        newcopy.isvalid = self.isvalid
        newcopy.actionprob = self.actionprob.copy()
        return newcopy

    def validBoard(self):
        return self.isvalid

    def unique(self,list1):

        # initialize a null list
        unique_list = []

        # traverse for all elements
        for x in list1:
            # check if exists in unique_list or not
            if x not in unique_list:
                unique_list.append(x)
        # print list
        return unique_list

    def validTilePerùutation(self,tiles):
            tileOnBoardTempory=[TileBinaire(tile[0], tile[1]) for tile in tiles]
            if len(tileOnBoardTempory)<2:
                return True
            if len(self.unique(tiles)) != len(tiles):
                return False

            testcolor=True
            testshape=True
            color=tileOnBoardTempory[0].color
            shape=tileOnBoardTempory[0].shape
            for tile in tileOnBoardTempory:
                if (np.all(color!=tile.color) and testcolor):
                        testcolor=testcolor and False
                        break
                if (np.all(shape!=tile.shape)):
                        testshape=testshape and False
                        break

            return testcolor or testshape

    def setActionprob(self):
        self.actionprob=[]
        for x in range(-20,20):
            for y in range(-20,20):
                for direction in range(0,4):
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
                        for j in range(0,6):
                            tile1=[]
                            for k in range(0,j+1):
                                tile1.append([TileColor[color],0,x+k*dirx,y+k*diry])
                            self.actionprob.append(tile1)
                    for shape in TileShape:
                        for j in range(0,6):
                            tile1=[]
                            for k in range(0,j+1):
                                tile1.append([0,TileShape[shape],x+k*dirx,y+k*diry])
                            self.actionprob.append(tile1)

    def getTile(self, position: Coordinate, tileOnBoardTempory):
        gettile = Tile(0, 0, Coordinate(0, 0))
        for tile in tileOnBoardTempory:
            if (tile.coordinate.x == position.x and tile.coordinate.y == position.y):
                return tile





















    def permutationFromPositionTiletileleft(self, permutation, posxtile, posytile):
        for j in range(0,4):
            if j==0:
                posx=posxtile+1
                posy=posytile
            if j==1:
                posx=posxtile-1
                posy=posytile
            if j==2:
                posx=posxtile
                posy=posytile-1
            if j==3:
                posx=posxtile
                posy=posytile+1

            if not self.tileOnBoard.ocupied(Coordinate(posx,posy)):
                for indx in range(0, 4):
                    if indx == 0:
                        sensx =-1
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

                    self.deepBoardbinarryCopy()
                    if self.testplaceTile(permutation, posx, posy,sensx,sensy) :
                        rackValidMove = [TileBinaire(tile[0], tile[1], Coordinate(posx +sensx*index , posy+sensy*index)) for index, tile in
                                         enumerate(permutation)]
                        self.listValidMoves.append(rackValidMove)



    def testplaceTile(self, permutation, posx, posy,sensx,sensy):




        for index,tile in enumerate(permutation):

            isvalidTempory=self.placetempory(TileBinaire(tile[0], tile[1]), posx +sensx* index, posy+sensy*index)
            if isvalidTempory == False:
                break
        return isvalidTempory

    def deepBoardbinarryCopy(self):
        self.tileOnBoardTempory=self.tileOnBoard.copy()




    def listValidMovePlayer2(self):
        self.listValidMoves=[]
        inp_list = self.player2.getRack()
        permutations = []
        for i in range(1,len(inp_list)+1):
            permutations.extend(list(itertools.permutations(inp_list, r=i)))

        if (len(permutations)>1):
            permutations =np.unique(permutations)
            vfunc = np.vectorize(self.validTilePerùutation)

            permutations = permutations[np.where(vfunc(permutations) == True)]

        if len(self.tileOnBoard.tiles) > 0:


                    [[self.permutationFromPositionTiletileleft(permut, tilebinairy.coordinate.x,
                                                             tilebinairy.coordinate.y) for permut in permutations] for tilebinairy in self.tileOnBoard.tiles]
        else:
            i = 0
            while i < len(permutations):
                self.deepBoardbinarryCopy()

                isvalidTempory = True
                for index, tile in enumerate(permutations[i]):
                    isvalidTempory = isvalidTempory and self.placetempory(TileBinaire(tile[0], tile[1]),
                                                                          0, 0)
                if isvalidTempory:
                    rackValidMove = [TileBinaire(tile[0], tile[1], Coordinate(0, 0)) for index, tile
                                     in
                                     enumerate(permutations[i])]
                    self.listValidMoves.add(tuple(rackValidMove))
                i += 1

    def listValidMovePlayer1(self):
        self.listValidMoves=[]
        inp_list = self.player1.getRack()
        permutations = []
        for i in range(1,len(inp_list)+1):
            permutations.extend(list(itertools.permutations(inp_list, r=i)))

        if (len(permutations)>1):
            permutations =np.unique(permutations)
            vfunc = np.vectorize(self.validTilePerùutation)

            permutations = permutations[np.where(vfunc(permutations) == True)]


        if len(self.tileOnBoard.tiles)>0:
            for tilebinairy in self.tileOnBoard.tiles:
                # test = np.vectorize(self.permutationFromPositionTiletileleft)
                # test(np.array(permutations), tilebinairy.coordinate.x,
                #      tilebinairy.coordinate.y, 1)
                for permut in permutations:
                    self.permutationFromPositionTiletileleft(permut, tilebinairy.coordinate.x,
                                                          tilebinairy.coordinate.y)
        else:
            i = 0
            while i < len(permutations):
                self.deepBoardbinarryCopy()
                isvalidTempory = True
                for index, tile in enumerate(permutations[i]):
                    isvalidTempory = isvalidTempory and self.placetempory(TileBinaire(tile[0], tile[1]), 0,0)

                if isvalidTempory:
                    rackValidMove = [TileBinaire(tile[0], tile[1], Coordinate(0, 0)) for index, tile
                                     in
                                     enumerate(permutations[i])]
                    self.listValidMoves.add(tuple(rackValidMove))
                i += 1