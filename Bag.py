# %%

import random
import itertools


from TileColor import TileColor
from TileShape import TileShape
import json




class Coordinate:
    def __init__(self,x:int,y:int):
        self.x = x
        self.y = y
class Tile:
    def __init__(self,color:TileColor,shape:TileShape,coordinate:Coordinate):
        self.color = color
        self.shape = shape
        self.coordinate =coordinate
        
    def get(self):
        return [self.color,self.shape,[self.coordinate.x,self.coordinate.y]]      
    def gettype(self):
        return [list(TileColor.keys())[self.color-1],list(TileShape.keys())[self.shape-1],[self.coordinate.x,self.coordinate.y]]


class TileOnBag:
    def __init__(self, index:int,tile:Tile):
        self.tile = tile 
        self.index= index

    def get(self):
        return [self.index,self.tile.get()]   


class TileOnBoard:
    def __init__(self, tile:Tile):
        self.tile = tile 
        

    def get(self):
        return [self.tile.get()]   

class Rack:

    def __init__(self):
        self.rack=[]
        
    def getTile(self,index):
        return self.rack[index]
    def isEmpty(self):
        return len(self.rack) == 0

class Bag:

    def __init__(self):
        self.bag = []

        for i in range(1,4):
            for color in TileColor:
                for shape in TileShape:

                    tile=Tile(color, shape,Coordinate(0,0))
                    self.bag.append(tile)

    def getTile(self,index):
        return self.bag[index]
    def isEmpty(self):
        return len(self.bag) == 0
    def getRamdomTile(self):
        randomIndex=random.randrange(0, len(self.bag), 2)
        tileRandom=Tile(self.bag[randomIndex].color, self.bag[randomIndex].shape,Coordinate(0,0))
        del self.bag[randomIndex]
        return  tileRandom

    def copy(self):
        bagcopy=Bag()
        bagcopy.bag=[]
        for bg in self.bag:
            bagcopy.bag.append(Tile(bg.color, bg.shape,bg.coordinate))
        return bagcopy


class Player:

  def __init__(self):
        self.point = 0
        self.rack = []
  def addTileToRack(self,bag:Bag):
       while (len(self.rack)<=5 and len(bag.bag)>0):
         self.rack.append(bag.getRamdomTile())
  def getRack(self):
    return [i.get() for i in self.rack]
        
  def delRack(self,tile:Tile):
    for rackdel in self.rack:
      if rackdel.shape == tile.shape and rackdel.color == tile.color:
        self.rack.remove(rackdel)
        break
  def newRack(self,bag):
      for rack in self.rack:
          bag.bag.append(rack)
      self.rack=[]
      self.addTileToRack(bag)

  def copy(self):
      player=Player()
      player.rack=[]
      for l in self.getRack():
          player.rack.append(Tile(l[0],l[1],Coordinate(l[2][0],l[2][1])))
      player.point=self.point

      return player

def makeStr(listValidMoves):
    newList=''
    for tiles in listValidMoves:
        newList=newList+str(tiles.get())
    return newList


def deepBoardCopy(tileOnBoard):
    tilesList=[]
    for tile in tileOnBoard:
        tilesList.append(Tile(tile.color,tile.shape,tile.coordinate))
    return tilesList


class Game:
    def __init__(self):
        self.bag = Bag()
        self.player1 = Player()
        self.player2 = Player()
        self.tileOnBoard=[]
        self.tileOnBoardTempory=[]
        self.listValidMoves=[]
        self.player1.addTileToRack(self.bag)
        self.player2.addTileToRack(self.bag)
        self.isvalid =True
        self.actionprob=[]

    def __copy__(self):
        newcopy = Game()
        newcopy.bag=self.bag.copy()
        newcopy.player1=self.player1.copy()
        newcopy.player2 = self.player2.copy()
        newcopy.tileOnBoard=self.tileOnBoard.copy()
        newcopy.tileOnBoardTempory = self.tileOnBoardTempory.copy()
        newcopy.listValidMoves = self.listValidMoves.copy()
        newcopy.isvalid = self.isvalid
        newcopy.actionprob = self.actionprob.copy()
        return newcopy
        
    def addTilesToBoardTempory(self,tile:Tile):
           
            self.tileOnBoardTempory.append(tile)
            if self.validBoard(self.tileOnBoardTempory):
                return self.tileOnBoardTempory
            return None  

    def permutationFromPositionTile(self,permutations,posx,posy):


          i=0
          dec=1
          if (posx==0 and posy==0 and len(self.tileOnBoard)<1):
              dec=0

          while i<len(permutations):
              rackValidMove = []
              self.tileOnBoardTempory = []
              self.tileOnBoardTempory = deepBoardCopy(self.tileOnBoard)
              for index,tile in enumerate(permutations[i]):

                  self.tileOnBoardTempory.append(Tile(tile[0], tile[1],Coordinate(posx,posy+index+dec)))
                  rackValidMove.append(Tile(tile[0], tile[1],Coordinate(posx,posy+index+dec)))
                  if self.validBoard(self.tileOnBoardTempory):
                      test=True
                  else:
                      test=False
                      break

              if test:
                  doublon = True
                  if (len(self.listValidMoves) > 0):
                      for tiles in self.listValidMoves:
                        if (makeStr(tiles) == makeStr(rackValidMove)):
                            doublon=False
                  if doublon :
                      self.listValidMoves.append(rackValidMove)


              i+=1
          i = 0
          while i < len(permutations):
              rackValidMove = []
              self.tileOnBoardTempory = []
              self.tileOnBoardTempory = deepBoardCopy(self.tileOnBoard)
              for index, tile in enumerate(permutations[i]):
                  self.tileOnBoardTempory.append(Tile(tile[0], tile[1], Coordinate(posx+ index+dec, posy )))
                  rackValidMove.append(Tile(tile[0], tile[1], Coordinate(posx+ index+dec, posy )))
                  if self.validBoard(self.tileOnBoardTempory):
                      test=True
                  else:
                      test=False
                      break
              if test:
                  doublon = True
                  if (len(self.listValidMoves) > 0):
                      for tiles in self.listValidMoves:
                          if (makeStr(tiles) == makeStr(rackValidMove)):
                              doublon = False
                  if doublon:
                      self.listValidMoves.append(rackValidMove)
              i += 1
          i = 0
          while i < len(permutations):
              rackValidMove = []
              self.tileOnBoardTempory = []
              self.tileOnBoardTempory = deepBoardCopy(self.tileOnBoard)
              for index, tile in enumerate(permutations[i]):
                  self.tileOnBoardTempory.append(Tile(tile[0], tile[1], Coordinate(posx, posy - index-dec)))
                  rackValidMove.append(Tile(tile[0], tile[1], Coordinate(posx, posy - index-dec)))
                  if self.validBoard(self.tileOnBoardTempory):
                      test=True
                  else:
                      test=False
                      break
              if test:
                  doublon = True
                  if (len(self.listValidMoves) > 0):
                      for tiles in self.listValidMoves:
                          if (makeStr(tiles) == makeStr(rackValidMove)):
                              doublon = False
                  if doublon:
                      self.listValidMoves.append(rackValidMove)
              i += 1
          i = 0
          while i < len(permutations):
              rackValidMove = []
              self.tileOnBoardTempory = []
              self.tileOnBoardTempory = deepBoardCopy(self.tileOnBoard)
              for index, tile in enumerate(permutations[i]):
                  self.tileOnBoardTempory.append(Tile(tile[0], tile[1], Coordinate(posx - index-dec, posy)))
                  rackValidMove.append(Tile(tile[0], tile[1], Coordinate(posx - index-dec, posy)))
                  if self.validBoard(self.tileOnBoardTempory):
                      test=True
                  else:
                      test=False
                      break
              if test:
                  doublon = True
                  if (len(self.listValidMoves) > 0):
                      for tiles in self.listValidMoves:
                          if (makeStr(tiles) == makeStr(rackValidMove)):
                              doublon = False
                  if doublon:
                      self.listValidMoves.append(rackValidMove)
              i += 1









    def listValidMovePlayer1(self):
        self.listValidMoves=[]
        inp_list = self.player1.getRack()
        permutations = []
        for i in range(1,len(inp_list)+1):
            permutations.extend(list(itertools.permutations(inp_list, r=i)))
        for permute in permutations:
            test=False
            if len(permute)>1:
                for k in range(0,len(permute)-1):
                    if permute[k][0]!=permute[k+1][0] and permute[k][1]!=permute[k+1][1]:
                        test=True
                    if permute[k][0]==permute[k+1][0] and permute[k][1]==permute[k+1][1]:
                        test=True
            if test:
                permutations.remove(permute)
        new_list = [] 
        for i in permutations : 
            if i not in new_list: 
                new_list.append(i)

        permutations=[]
        for tiles in new_list:
            if (self.validTilePerùutation([Tile(tile[0],tile[1],Coordinate(0,0)) for tile in tiles])):
                permutations.append(tiles)

        if len(self.tileOnBoard)>0:
            for tile in self.tileOnBoard:
                self.permutationFromPositionTile(permutations,tile.coordinate.x,tile.coordinate.y)
        else:
            self.permutationFromPositionTile(permutations, 0,0)



    def listValidMovePlayer2(self):
        self.listValidMoves=[]
        inp_list = self.player2.getRack()
        permutations = []
        for i in range(1,len(inp_list)+1):
            permutations.extend(list(itertools.permutations(inp_list, r=i)))
        for permute in permutations:
            test=False
            if len(permute)>1:
                for k in range(0,len(permute)-1):
                    if permute[k][0]!=permute[k+1][0] and permute[k][1]!=permute[k+1][1]:
                        test=True
                    if permute[k][0]==permute[k+1][0] and permute[k][1]==permute[k+1][1]:
                        test=True

            if test:
                permutations.remove(permute)
        new_list = [] 
        for i in permutations : 
            if i not in new_list: 
                new_list.append(i)

        permutations = []
        for tiles in new_list:
            if (self.validTilePerùutation([Tile(tile[0],tile[1],Coordinate(0,0)) for tile in tiles])):
                permutations.append(tiles)

        if len(self.tileOnBoard) > 0:
            for tile in self.tileOnBoard:
                self.permutationFromPositionTile(permutations, tile.coordinate.x, tile.coordinate.y)
        else:
            self.permutationFromPositionTile(permutations, 0, 0)


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
                                tile1.append([TileColor[color],0,direction,x+k*dirx,y+k*diry])
                            self.actionprob.append(tile1)
                    for shape in TileShape:
                        for j in range(0,6):
                            tile1=[]
                            for k in range(0,j+1):
                                tile1.append([0,TileShape[shape],direction,x+k*dirx,y+k*diry])
                            self.actionprob.append(tile1)

    def playPlayer1(self):
        
        self.tileOnBoardTempory=deepBoardCopy(self.tileOnBoard)
        self.listValidMovePlayer1()
        self.tileOnBoard=[]
        for tile in self.listValidMoves[len(self.listValidMoves)-1]:
            self.tileOnBoard.append(tile)
            self.player1.delRack(tile)
            self.player1.addTileToRack(self.bag)
        
        self.tileOnBoardTempory=[] 

    def playPlayer2(self):
        
        self.tileOnBoardTempory=deepBoardCopy(self.tileOnBoard)
        self.listValidMovePlayer2()
        self.tileOnBoard=[]
        for tile in self.listValidMoves[len(self.listValidMoves)-1]:
            self.tileOnBoard.append(tile)
            self.player2.delRack(tile)
            self.player2.addTileToRack(self.bag)
        
        self.tileOnBoardTempory=[]

    def gettileOnBoardTempory(self):
        tilesBoard=[]   
        for tile in self.tileOnBoardTempory:
            tilesBoard.append(tile)
        return tilesBoard 


    def getBoard(self):
        tilesBoard=[]   
        for tile in self.tileOnBoard:
            tilesBoard.append([[tile.color,tile.shape],[tile.coordinate.x,tile.coordinate.y]])
        return tilesBoard

    def getTile(self,position:Coordinate,tileOnBoardTempory):
        gettile=Tile(0,0,Coordinate(0,0))
        for tile in tileOnBoardTempory:
            if (tile.coordinate.x==position.x and tile.coordinate.y==position.y):
                 return tile

        return gettile
    def goodPosition(self,position:Coordinate):
        for tile in self.tileOnBoard:
            if (tile.coordinate.x==position.x and tile.coordinate.y==position.y):
                return False
        return True
    def goodPositionTempory(self,tileOnBoardTempory):
        for index in range(len(tileOnBoardTempory)):
            for indexOther in range(len(tileOnBoardTempory)):
                if index!=indexOther:
                    if (tileOnBoardTempory[index].coordinate.x==tileOnBoardTempory[indexOther].coordinate.x and tileOnBoardTempory[index].coordinate.y==tileOnBoardTempory[indexOther].coordinate.y):
                        return False
        return True
    def validBoard(self,tileOnBoardTempory):

            if len(tileOnBoardTempory)<2:
                    return True
            if (self.goodPositionTempory(tileOnBoardTempory)):

                for tile in tileOnBoardTempory:
                    for pos in range (1,6):
                        tileup = self.getTile(Coordinate(tile.coordinate.x, tile.coordinate.y - pos), tileOnBoardTempory)
                        if (tileup.color!=0):
                            if (tile.color!=tileup.color and tile.shape!=tileup.shape):
                                return False
                            if (tileup.color==tile.color and tileup.shape==tile.shape):
                                return False
                        else:
                            break
                for tile in tileOnBoardTempory:
                    for pos in range(1, 6):
                        tiledown = self.getTile(Coordinate(tile.coordinate.x, tile.coordinate.y + pos), tileOnBoardTempory)

                        if (tiledown.color!=0):
                            if (tile.color!=tiledown.color and tile.shape!=tiledown.shape):
                                return False
                            if tile.color == tiledown.color and tile.shape == tiledown.shape:
                                return False
                        else:
                            break
                for tile in tileOnBoardTempory:
                    for pos in range(1, 6):
                        tileleft = self.getTile(Coordinate(tile.coordinate.x - pos, tile.coordinate.y), tileOnBoardTempory)

                        if (tileleft.color!=0):
                            if (tile.color!=tileleft.color and tile.shape!=tileleft.shape):
                                return False
                            if tile.color == tileleft.color and tile.shape == tileleft.shape:
                                return False
                        else:
                            break
                for tile in tileOnBoardTempory:
                    for pos in range(1, 6):
                        tileright=self.getTile(Coordinate(tile.coordinate.x+pos,tile.coordinate.y),tileOnBoardTempory)

                        if (tileright.color!=0):
                            if (tile.color!=tileright.color and tile.shape!=tileright.shape):
                                return False
                            if tile.color == tileright.color and tile.shape == tileright.shape:
                                return False
                        else:
                            break

                for tile in tileOnBoardTempory:
                    tileup=self.getTile(Coordinate(tile.coordinate.x,tile.coordinate.y-1),tileOnBoardTempory)
                    tiledown=self.getTile(Coordinate(tile.coordinate.x,tile.coordinate.y+1),tileOnBoardTempory)
                    tileleft=self.getTile(Coordinate(tile.coordinate.x-1,tile.coordinate.y),tileOnBoardTempory)
                    tileright=self.getTile(Coordinate(tile.coordinate.x+1,tile.coordinate.y),tileOnBoardTempory)
                    tileSE = self.getTile(Coordinate(tile.coordinate.x+1, tile.coordinate.y + 1), tileOnBoardTempory)
                    tileSO = self.getTile(Coordinate(tile.coordinate.x-1, tile.coordinate.y+ 1), tileOnBoardTempory)
                    tileNE = self.getTile(Coordinate(tile.coordinate.x + 1, tile.coordinate.y-1), tileOnBoardTempory)
                    tileNO = self.getTile(Coordinate(tile.coordinate.x - 1, tile.coordinate.y-1), tileOnBoardTempory)




                    if (tile.color!=0 and tileright.color==0 and tiledown.color==0 and tileleft.color==0 and tileup.color==0):
                        return False
                    if (tile.color!=0 and tileright.color==0 and tiledown.color==0 and tileSE.color!=0):
                        return False
                    if (tile.color!=0 and tileright.color==0 and tileup.color==0 and tileNE.color!=0):
                        return False
                    if (tile.color!=0 and tileleft.color==0 and tiledown.color==0 and tileSO.color!=0):
                        return False
                    if (tile.color!=0 and tileleft.color==0 and tileup.color==0 and tileNO.color!=0):
                        return False






                    if (tileup.color!=0 and tiledown.color!=0):
                        if ((tile.color==tileup.color and tile.color==tiledown.color) and (tile.shape==tileup.shape and tile.shape==tiledown.shape)):
                            return False


                    if (tileright.color!=0 and tileleft.color!=0):
                        if ((tile.color==tileright.color and tile.color==tileleft.color) and (tile.shape==tileright.shape and tile.shape==tileleft.shape)):
                            return False


                    if (tileup.color==0 and tiledown.color==0 and tileleft.color==0 and tileright.color==0):
                        return False




                return True


            return False

    def validTilePerùutation(self,tileOnBoardTempory):

            if len(tileOnBoardTempory)<2:
                return True
            testcolor=True
            testshape=True
            color=tileOnBoardTempory[0].color
            shape=tileOnBoardTempory[0].shape
            for tile in tileOnBoardTempory:
                if (color!=tile.color):
                        testcolor=testcolor and False
                        break
            for tile in tileOnBoardTempory:
                if (shape!=tile.shape):
                        testshape=testshape and False
                        break

            return testcolor or testshape





