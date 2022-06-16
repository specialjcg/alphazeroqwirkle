# %%
TileColor={'Green':1,'Blue':2,'Purple':3,'Red':4,'Orange':5,'Yellow':6}
TileShape={'Circle':1,'Square':2,'Diamond':3,'Clover':4,'FourPointStar':5,'EightPointStar':6}
import random
import itertools
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
        j=1
        for i in range(1,4):
            for color in TileColor:
                for shape in TileShape:

                    tile=TileOnBag(j,Tile(color, shape,Coordinate(0,0)))
                    self.bag.append(tile)
                    j+=1
    def getTile(self,index):
        return self.bag[index]
    def isEmpty(self):
        return len(self.bag) == 0
    def getRamdomTile(self):
        randomIndex=random.randrange(0, len(self.bag), 2)
        tileRandom=Tile(self.bag[randomIndex].tile.color, self.bag[randomIndex].tile.shape,Coordinate(0,0))
        del self.bag[randomIndex]
        return  tileRandom
        

    


# %%
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


  



# %%
from collections import OrderedDict

# %%
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
          
        
    def addTilesToBoardTempory(self,tile:Tile):
           
            self.tileOnBoardTempory.append(tile)
            if self.validBoard():
                return self.tileOnBoardTempory
            return None  

    def permutationFromPositionTile(self,permutations,posy,posx):
        if self.goodPosition(Coordinate(posx,posy)):
          for i in range(0,len(permutations)):
                    temporyBoard=self.tileOnBoardTempory.copy()
                    test=True
                    for x in range(0,len(permutations[i])):
                        val2=self.addTilesToBoardTempory(Tile(permutations[i][x][0], permutations[i][x][1],Coordinate(posx,posy+x)))
                        if (val2!=None):
                            test=test and True
                        else:
                            test=False
                            break                                
                    if (test):
                        self.listValidMoves.append(self.tileOnBoardTempory)
                    self.tileOnBoardTempory=temporyBoard.copy()
          for i in range(0,len(permutations)):
                    temporyBoard=self.tileOnBoardTempory.copy()
                    test=True
                    for x in range(0,len(permutations[i])):
                        val2=self.addTilesToBoardTempory(Tile(permutations[i][x][0], permutations[i][x][1],Coordinate(posx+x,posy)))
                        if (val2!=None):
                            test=test and True
                        else:
                            test=False
                            break                                
                    if (test):
                        self.listValidMoves.append(self.tileOnBoardTempory)
                    self.tileOnBoardTempory=temporyBoard.copy()
          for i in range(0,len(permutations)):
                    temporyBoard=self.tileOnBoardTempory.copy()
                    test=True
                    for x in range(0,len(permutations[i])):
                        val2=self.addTilesToBoardTempory(Tile(permutations[i][x][0], permutations[i][x][1],Coordinate(posx-x,posy)))
                        if (val2!=None):
                            test=test and True
                        else:
                            test=False
                            break                                
                    if (test):
                        self.listValidMoves.append(self.tileOnBoardTempory)
                    self.tileOnBoardTempory=temporyBoard.copy()
          for i in range(0,len(permutations)):
                    temporyBoard=self.tileOnBoardTempory.copy()
                    test=True
                    for x in range(0,len(permutations[i])):
                        val2=self.addTilesToBoardTempory(Tile(permutations[i][x][0], permutations[i][x][1],Coordinate(posx,posy-x)))
                        if (val2!=None):
                            test=test and True
                        else:
                            test=False
                            break                                
                    if (test):
                        self.listValidMoves.append(self.tileOnBoardTempory)
                    self.tileOnBoardTempory=temporyBoard.copy()            

    def permutePlayer1(self):
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
        
            if test:
                permutations.remove(permute)
        new_list = [] 
        for i in permutations : 
            if i not in new_list: 
                new_list.append(i)

        permutations=new_list.copy()
        if len(self.tileOnBoard)>1:    
            for tileboard in self.tileOnBoard:
                posxup=tileboard.coordinate.x
                posyup=tileboard.coordinate.y-1
                self.permutationFromPositionTile(permutations,posyup,posxup)
            for tileboard in self.tileOnBoard:
                posxup=tileboard.coordinate.x
                posyup=tileboard.coordinate.y+1
                self.permutationFromPositionTile(permutations,posyup,posxup)
            for tileboard in self.tileOnBoard:
                posxup=tileboard.coordinate.x-1
                posyup=tileboard.coordinate.y
                self.permutationFromPositionTile(permutations,posyup,posxup)
            for tileboard in self.tileOnBoard:
                posxup=tileboard.coordinate.x+1
                posyup=tileboard.coordinate.y
                self.permutationFromPositionTile(permutations,posyup,posxup)            
        else:
            self.permutationFromPositionTile(permutations,0,0)

    def permutePlayer2(self):
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
        
            if test:
                permutations.remove(permute)
        new_list = [] 
        for i in permutations : 
            if i not in new_list: 
                new_list.append(i)

        permutations=new_list.copy()
        if len(self.tileOnBoard)>1:    
            for tileboard in self.tileOnBoard:
                posxup=tileboard.coordinate.x
                posyup=tileboard.coordinate.y-1
                self.permutationFromPositionTile(permutations,posyup,posxup)
            for tileboard in self.tileOnBoard:
                posxup=tileboard.coordinate.x
                posyup=tileboard.coordinate.y+1
                self.permutationFromPositionTile(permutations,posyup,posxup)
            for tileboard in self.tileOnBoard:
                posxup=tileboard.coordinate.x-1
                posyup=tileboard.coordinate.y
                self.permutationFromPositionTile(permutations,posyup,posxup)
            for tileboard in self.tileOnBoard:
                posxup=tileboard.coordinate.x+1
                posyup=tileboard.coordinate.y
                self.permutationFromPositionTile(permutations,posyup,posxup)            
        else:
            self.permutationFromPositionTile(permutations,0,0)

    def setActionprob(self):        
        for x in range(-20,20):
            for y in range(-20,20):
                for direction in range(0,4):
                    for color in TileColor:
                        for j in range(0,6):
                            tile1=[]
                            for k in range(0,j+1):
                                
                                tile1.append([TileColor[color],0,direction,x,y])          
                            self.actionprob.append(tile1)
                    for shape in TileShape:
                        for j in range(0,6):
                            tile1=[]
                            for k in range(0,j+1):
                                tile1.append([0,TileShape[shape],direction,x,y])
                            self.actionprob.append(tile1)         

    def playPlayer1(self):
        
        self.tileOnBoardTempory=self.tileOnBoard.copy()
        self.permutePlayer1()
        self.tileOnBoard=[]
        for tile in self.listValidMoves[len(self.listValidMoves)-1]:
            self.tileOnBoard.append(tile)
            self.player1.delRack(tile)
            self.player1.addTileToRack(self.bag)
        
        self.tileOnBoardTempory=[] 

    def playPlayer2(self):
        
        self.tileOnBoardTempory=self.tileOnBoard.copy()
        self.permutePlayer2()
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

    def getTile(self,position:Coordinate):
        gettile=Tile(0,0,Coordinate(0,0));
        for tile in self.tileOnBoardTempory:
            if (tile.coordinate.x==position.x and tile.coordinate.y==position.y):
                 return tile

        return gettile
    def goodPosition(self,position:Coordinate):
        for tile in self.tileOnBoard:
            if (tile.coordinate.x==position.x and tile.coordinate.y==position.y):
                return False
        return True        
    def validBoard(self):
        if len(self.tileOnBoardTempory)<2:
            return True
        test=True
    
        for tile in self.tileOnBoardTempory:
            tileup=self.getTile(Coordinate(tile.coordinate.x,tile.coordinate.y-1))
            tiledown=self.getTile(Coordinate(tile.coordinate.x,tile.coordinate.y+1))
            tileleft=self.getTile(Coordinate(tile.coordinate.x-1,tile.coordinate.y))
            tileright=self.getTile(Coordinate(tile.coordinate.x+1,tile.coordinate.y))

            if (tileup.color!=0):
                if (tile.color!=tileup.color and tile.shape!=tileup.shape):
                    test=test and False
                    break
            if (tiledown.color!=0):    
                if (tile.color!=tiledown.color and tile.shape!=tiledown.shape):
                    test=test and False
                    break
            if (tileleft.color!=0):    
                if (tile.color!=tileleft.color and tile.shape!=tileleft.shape):
                    test=test and False
                    break
            if (tileright.color!=0):   
                if (tile.color!=tileright.color and tile.shape!=tileright.shape):
                    test=test and False
                    break

            if (tileup.color!=0 and tiledown.color!=0):           
                if ((tile.color==tileup.color and tile.color==tiledown.color) and (tile.shape==tileup.shape and tile.shape==tiledown.shape)):
                    test=test and True
                else:
                    test=test and False
                    break    
            if (tileright.color!=0 and tileleft.color!=0):
                if ((tile.color==tileright.color and tile.color==tileleft.color) and (tile.shape==tileright.shape and tile.shape==tileleft.shape)):
                    test=test and True
                else:
                    test=test and False 
                    break       
            if (tileup.color==0 and tiledown.color==0 and tileleft.color==0 and tileright.color==0):
                test=test and False
                break 

        return test
           
    

    


# %%

game=Game()
print(game.player1.getRack())
game.playPlayer1()
print(game.getBoard())
print(game.player1.getRack())


# %%

# game.playPlayer1()
# print(game.getBoard())




