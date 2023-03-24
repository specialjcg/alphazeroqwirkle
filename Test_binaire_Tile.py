import itertools
from datetime import time
from random import randrange
from timeit import timeit

import numpy as np

from Bag import Game, Tile, Coordinate
from BagBinaire import BagBinaire
from BagNumpy import BagNumpy
from GameBinary import GameBinaire
from GameNumpy import GameNumpy
from PlayerBinaire import PlayerBinaire
from PlayerNumpy import PlayerNumpy
from TileBinaire import TileBinaire
from TileColor import TileColor
from TileShape import TileShape
from qwirckleAlphazero import boardPlayToGridNorm, gridNormToBoardPlay, convertToBoard, gridNormtoRack, deepGridCopy, \
    findindexinActionprob, get_valid_moves, findindexinActionprobnumpy


class TestClassBinaireDemoInstance:

    def test_tileBinaire(self):
        tile = TileBinaire(0,0,Coordinate(0,0))
        assert tile.color == 0
        assert tile.shape ==  0

    def test_tilenumpy(self):
        tilecolor = np.zeros(shape=(108,108), dtype=np.int32)
        print( tilecolor[54,54])
        assert tilecolor[54,54]==0

    def test_tileGreenCircleBinaire(self):
        tile = TileBinaire(1, 1,Coordinate(0,0))
        assert tile.color == 1
        assert tile.shape == 1

    def test_tileGreenCirclenumpy(self):
        game=GameNumpy()

        game.place(1,1,0,0)
        assert game.tilecolor[54,54]== 1
        assert game.tileshape[54,54]== 1

    def test_tileBagBinaire(self):
        bag = BagNumpy()
        assert bag.getTile(10)==(2, 5)
        assert bag.bagCount()==108


    def test_playerBinaire(self):
        bag = BagBinaire()
        player1=PlayerBinaire()
        player1.addTileToRack(bag)
        assert len(player1.rack) == 6
        assert  len(bag.bag)==102

    def test_playerNumpy(self):
        bag = BagNumpy()
        player1 = PlayerNumpy()
        player1.addTileToRack(bag)
        assert player1.rackCount() == 6
        assert bag.bagCount() == 102
    def test_playerNumpynewrack(self):
        bag = BagNumpy()
        player1 = PlayerNumpy()
        player1.addTileToRack(bag)
        player1.newRack(bag)
        assert player1.rackCount() == 6
        assert bag.bagCount() == 102
    def test_one(self):
        game = GameBinaire()
        assert len(game.bag.bag) == 96

    def test_oneNumpy(self):
        game = GameNumpy()
        assert game.bag.bagCount() == 96

    def test_tileOnBoard_False_for_same_place(self):
        game = GameBinaire()
        tile1 = TileBinaire(1,1)
        tile2=TileBinaire(1,2)
        game.isvalid=game.place(tile1,0,0)
        game.isvalid=game.place(tile2,0,0)

        assert game.validBoard() == False

    def test_tileOnBoard_False_for_same_placeNumpy(self):
        game = GameNumpy()

        game.isvalid = game.place(1,1, 0, 0)
        game.isvalid = game.place(1,2, 0, 0)

        assert game.validBoard() == False
    def test_tileOnBoard_False_for_same_Tile_neibhour(self):
        game = GameBinaire()
        tile1 = TileBinaire(1, 1)
        tile2 = TileBinaire(1, 1)
        game.isvalid=game.place(tile1, 0, 0)
        game.isvalid=game.place(tile2, 0, 1)

        assert game.validBoard() == False

    def test_tileOnBoard_False_for_same_Tile_neibhourNumpy(self):
        game = GameNumpy()

        game.isvalid = game.place(1,1, 0, 0)
        game.isvalid = game.place(1,1, 0, 1)

        assert game.validBoard() == False
    def test_tileOnBoard_True_for_diff_Tile_neibhour(self):
        game = GameBinaire()
        tile1 = TileBinaire(1, 1)
        tile2 = TileBinaire(2, 1)
        game.isvalid=game.place(tile1, 0, 0)
        game.isvalid=game.place(tile2, 1, 0)

        assert game.validBoard() == True

    def test_tileOnBoard_True_for_diff_Tile_neibhourNumpy(self):
        game = GameNumpy()
        game.isvalid=game.place(1,1, 0, 0)
        game.isvalid=game.place(2,1, 1, 0)
        assert game.validBoard() == True

    def test_tileOnBoard_True_for_diff_3_Tile_neibhour(self):
        game = GameBinaire()
        game.isvalid=game.place(TileBinaire(1, 1), 0, 0)
        game.isvalid=game.place(TileBinaire(2, 1), 1, 0)
        game.isvalid=game.place(TileBinaire(3, 1), 2, 0)

        assert game.validBoard() == True

    def test_tileOnBoard_True_for_diff_3_Tile_neibhourNumpy(self):
        game =  GameNumpy()
        game.isvalid = game.place(1, 1, 0, 0)
        game.isvalid = game.place(2, 1, 1, 0)
        game.isvalid = game.place(3, 1, 2, 0)

        assert game.validBoard() == True

    def test_tileOnBoard_False_for_2_same_Tile_on_same_line(self):
        game = GameBinaire()
        game.isvalid=game.place(TileBinaire(1, 1), 0, 0)
        game.isvalid=game.place(TileBinaire(2, 1), 1, 0)
        game.isvalid=game.place(TileBinaire(1, 1), 2, 0)

        assert game.validBoard() == False

    def test_tileOnBoard_False_for_2_same_Tile_on_same_lineNumpy(self):
        game = GameNumpy()
        game.isvalid = game.place(1, 1, 0, 0)
        game.isvalid = game.place(2, 1, 1, 0)
        game.isvalid = game.place(1, 1, 2, 0)

        assert game.validBoard() == False

    def test_tileOnBoard_true_for_2_same_Tile_on_same_line_separate_by_space(self):
        game = GameBinaire()
        game.isvalid=game.place(TileBinaire(1, 1), 0, 0)
        game.isvalid=game.place(TileBinaire(2, 1), 1, 0)
        game.isvalid=game.place(TileBinaire(3, 1), 2, 0)
        game.isvalid=game.place(TileBinaire(4, 1), 0, 1)

        game.isvalid=game.place(TileBinaire(4, 1), 2, 1)
        import matplotlib.pyplot as plt
        import cv2

        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        from cairosvg import svg2png


        fig, ax = plt.subplots(figsize=(12, 8))

        for index,x in enumerate(game.board):
                svg2png(url="/home/jcgouleau/PycharmProjects/alphazeroqwirkle/img/" + x.color + x.shape + ".svg",
                        write_to="stinkbug.png")
                plt.xlim([0, 50])
                plt.ylim([0, 50])
                arr_img = plt.imread("stinkbug.png")
                half = cv2.resize(arr_img, (0, 0), fx=0.1, fy=0.1)
                im = OffsetImage(half)

                ab = AnnotationBbox(im, (25 + x.coordinate.x * 2.5, 25 + x.coordinate.y * 3.5), xycoords='data')
                ax.add_artist(ab)

        plt.show(block=True)
        plt.interactive(False)

        assert game.validBoard() == True

    def test_tileOnBoard_true_for_2_same_Tile_on_same_line_separate_by_spaceNumpy(self):
        game = GameNumpy()
        game.isvalid=game.place(1, 1, 0, 0)
        game.isvalid=game.place(2, 1, 1, 0)
        game.isvalid=game.place(3, 1, 2, 0)
        game.isvalid=game.place(4, 1, 0, 1)

        game.isvalid=game.place(4, 1, 2, 1)


        assert game.validBoard() == True

    def test_tileOnBoard_false_for_2_same_Tile_on_same_line(self):
        game = GameBinaire()
        game.isvalid=game.place(TileBinaire(1, 1), 0, 0)
        game.isvalid=game.place(TileBinaire(2, 1), 1, 0)
        game.isvalid=game.place(TileBinaire(3, 1), 2, 0)
        game.isvalid=game.place(TileBinaire(4, 1), 0, 1)

        game.isvalid=game.place(TileBinaire(4, 1), 1, 1)
        # import matplotlib.pyplot as plt
        # import cv2
        #
        # from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        # from cairosvg import svg2png
        #
        # fig, ax = plt.subplots(figsize=(12, 8))
        #
        # for index, x in enumerate(game.board):
        #     svg2png(url="/home/jcgouleau/PycharmProjects/alphazeroqwirkle/img/" + x.color + x.shape + ".svg",
        #             write_to="stinkbug.png")
        #     plt.xlim([0, 50])
        #     plt.ylim([0, 50])
        #     arr_img = plt.imread("stinkbug.png")
        #     half = cv2.resize(arr_img, (0, 0), fx=0.1, fy=0.1)
        #     im = OffsetImage(half)
        #
        #     ab = AnnotationBbox(im, (25 + x.coordinate.x * 2.5, 25 + x.coordinate.y * 3.5), xycoords='data')
        #     ax.add_artist(ab)
        #
        # plt.show(block=True)
        # plt.interactive(False)

        assert game.validBoard() == False

    def test_tileOnBoard_false_for_2_same_Tile_on_same_lineNumpy(self):
        game = GameNumpy()
        game.isvalid=game.place(1, 1, 0, 0)
        game.isvalid=game.place(2, 1, 1, 0)
        game.isvalid=game.place(3, 1, 2, 0)
        game.isvalid=game.place(4, 1, 0, 1)

        game.isvalid=game.place(4, 1, 1, 1)
        # import matplotlib.pyplot as plt
        # import cv2
        #
        # from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        # from cairosvg import svg2png
        #
        # fig, ax = plt.subplots(figsize=(12, 8))
        #
        # for index, x in enumerate(game.board):
        #     svg2png(url="/home/jcgouleau/PycharmProjects/alphazeroqwirkle/img/" + x.color + x.shape + ".svg",
        #             write_to="stinkbug.png")
        #     plt.xlim([0, 50])
        #     plt.ylim([0, 50])
        #     arr_img = plt.imread("stinkbug.png")
        #     half = cv2.resize(arr_img, (0, 0), fx=0.1, fy=0.1)
        #     im = OffsetImage(half)
        #
        #     ab = AnnotationBbox(im, (25 + x.coordinate.x * 2.5, 25 + x.coordinate.y * 3.5), xycoords='data')
        #     ax.add_artist(ab)
        #
        # plt.show(block=True)
        # plt.interactive(False)

        assert game.validBoard() == False
    def test_tilesetactionprob(self):
        game = GameBinaire()
        game.setActionprob()

        assert game.actionprob[236347] == [[0, 2, 0, 0], [0, 2,  -1, 0]]
        assert game.actionprob[236431] == [[0, 4, 0, 0], [0, 4,  0, -1]]

    def test_tilesetactionprobNumpy(self):
        game = GameNumpy()
        game.setActionprob()

        assert game.actionprob[236347] == [[0, 2, 0, 0], [0, 2,  -1, 0]]
        assert game.actionprob[236431] == [[0, 4, 0, 0], [0, 4,  0, -1]]

    def test_tilesetactionprobNumpytest(self):
        game = GameNumpy()
        import datetime
        start_time = datetime.datetime.now().time().strftime('%H:%M:%S.%f')
        game.setActionprobtest()
        end_time = datetime.datetime.now().time().strftime('%H:%M:%S.%f')
        total_time = (datetime.datetime.strptime(end_time, '%H:%M:%S.%f') - datetime.datetime.strptime(start_time,
                                                                                                       '%H:%M:%S.%f'))
        print('total_time:' + str(total_time))

        assert len(game.actionprob)==6156
        assert game.actionprob[25] == [[5, 2, -30, -30]]
        assert game.actionprob[6156] == [[4, 4, -14, 13], [4, 4, -15, 13]]

    def test_tilechangerack(self):
        game = GameBinaire()
        val = game.player1.getRack()

        game.player1.newRack(game.bag)
        newrack = game.player1.getRack()
        vallen = len(game.bag.bag)
        assert np.any(val != newrack)
        assert vallen == 96

    def test_tilechangerackNumpy(self):
        game = GameNumpy()
        val = game.player1.getRack()
        game.player1.newRack(game.bag)
        newrack = game.player1.getRack()
        vallen = game.bag.bagCount()
        assert np.any(np.array(val)!= np.array(newrack))
        assert vallen == 96

    def test_gamecopy(self):
        game = GameBinaire()
        game.setActionprob()
        newgame=game.__copy__()
        newgame.player1.newRack(newgame.bag)
        assert np.any(newgame.player1.getRack()!=game.player1.getRack())


        
    def test_listvalide_movesnotsamecolorandshapeonline(self):
        game = GameBinaire()
        game.setActionprob()
        import matplotlib.pyplot as plt
        import cv2

        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        from cairosvg import svg2png

        fig, ax = plt.subplots(figsize=(12, 8))
        TileShape = {'Circle': 1, 'Square': 2, 'Diamond': 3, 'Clover': 4, 'FourPointStar': 5, 'EightPointStar': 6}
        TileColor = {'Green': 1, 'Blue': 2, 'Purple': 3, 'Red': 4, 'Orange': 5, 'Yellow': 6}
        game.player2.rack=[]
        game.player2.rack.append(TileBinaire(3,1))
        game.player2.rack.append(TileBinaire(3,1))
        game.player2.rack.append(TileBinaire(5,1))
        game.player2.rack.append(TileBinaire(1,6))
        game.player2.rack.append(TileBinaire(3,6))
        game.player2.rack.append(TileBinaire(3,4))


        game.isvalid=game.place(TileBinaire(4,3),0,0)
        game.isvalid=game.place(TileBinaire(2,3),0,1)
        game.isvalid=game.place(TileBinaire(4,5),1,0)
        game.isvalid=game.place(TileBinaire(1,5),1,-1)
        game.isvalid=game.place(TileBinaire(5,3),-1,1)
        game.isvalid=game.place(TileBinaire(5,1),-2,1)



        import matplotlib.pyplot as plt
        import cv2

        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        from cairosvg import svg2png

        fig, ax = plt.subplots(figsize=(12, 8))

        for index, x in enumerate(game.board):
            svg2png(url="/home/jcgouleau/PycharmProjects/alphazeroqwirkle/img/" + x.color + x.shape + ".svg",
                    write_to="stinkbug.png")
            plt.xlim([0, 50])
            plt.ylim([0, 50])
            arr_img = plt.imread("stinkbug.png")
            half = cv2.resize(arr_img, (0, 0), fx=0.1, fy=0.1)
            im = OffsetImage(half)

            ab = AnnotationBbox(im, (25 + x.coordinate.x * 2.5, 25 + x.coordinate.y * 3.5), xycoords='data')
            ax.add_artist(ab)

        plt.show(block=True)
        plt.interactive(False)
        assert game.validBoard() == False
        
    def test_listvalide_movesnotsamecolorandshapeonlineNumpy(self):
        game = GameNumpy()
        game.setActionprob()
        import matplotlib.pyplot as plt

        TileShape = {'Circle': 1, 'Square': 2, 'Diamond': 3, 'Clover': 4, 'FourPointStar': 5, 'EightPointStar': 6}
        TileColor = {'Green': 1, 'Blue': 2, 'Purple': 3, 'Red': 4, 'Orange': 5, 'Yellow': 6}
        game.player2.zero()
        game.player2.addRack(3,1)
        game.player2.addRack(3,1)
        game.player2.addRack(5,1)
        game.player2.addRack(1,6)
        game.player2.addRack(3,6)
        game.player2.addRack(3,4)


        game.isvalid=game.place(4,3,0,0)
        game.isvalid=game.place(2,3,0,1)
        game.isvalid=game.place(4,5,1,0)
        game.isvalid=game.place(1,5,1,-1)
        game.isvalid=game.place(5,3,-1,1)
        game.isvalid=game.place(5,1,-2,1)



        import matplotlib.pyplot as plt
        import cv2

        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        from cairosvg import svg2png

        fig, ax = plt.subplots(figsize=(12, 8))

        for x in range(108):
            for y in range(108):
                if game.tilecolor[x,y]!=0:
                    svg2png(url="/home/jcgouleau/PycharmProjects/alphazeroqwirkle/img/" + list(TileColor.keys())[game.tilecolor[x,y]] + list(TileShape.keys())[game.tileshape[x,y]] + ".svg",
                            write_to="stinkbug.png")
                    plt.xlim([0, 50])
                    plt.ylim([0, 50])
                    arr_img = plt.imread("stinkbug.png")
                    half = cv2.resize(arr_img, (0, 0), fx=0.1, fy=0.1)
                    im = OffsetImage(half)

                    ab = AnnotationBbox(im, (25+ (x-54) * 2.5, 25+(y-54) * 3.5), xycoords='data')
                    ax.add_artist(ab)

        plt.show(block=True)
        plt.interactive(False)
        assert game.validBoard() == False

    def test_listvalide_movesnot_empty(self):
        game = GameBinaire()
        game.setActionprob()
        import matplotlib.pyplot as plt
        import cv2

        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        from cairosvg import svg2png


        TileShape = {'Circle': 1, 'Square': 2, 'Diamond': 3, 'Clover': 4, 'FourPointStar': 5, 'EightPointStar': 6}
        TileColor = {'Green': 1, 'Blue': 2, 'Purple': 3, 'Red': 4, 'Orange': 5, 'Yellow': 6}
        game.player2.rack = []
        game.player2.rack.append(TileBinaire(3, 1))
        game.player2.rack.append(TileBinaire(3, 1))
        game.player2.rack.append(TileBinaire(5, 1))
        game.player2.rack.append(TileBinaire(1, 6))
        game.player2.rack.append(TileBinaire(3, 6))
        game.player2.rack.append(TileBinaire(3, 4))

        game.isvalid = game.place(TileBinaire(4, 3), 0, 0)
        game.isvalid = game.place(TileBinaire(2, 3), 0, 1)
        game.isvalid = game.place(TileBinaire(4, 5), 1, 0)
        game.isvalid = game.place(TileBinaire(1, 5), 1, -1)
        game.isvalid = game.place(TileBinaire(5, 3), -1, 1)





        from timeit import default_timer as timer
        start = timer()


        game.listValidMovePlayer2()
        end = timer()
        print(end - start)



        assert len(game.listValidMoves)==10

    def test_listvalide_movesnot_empty(self):
        game = GameBinaire()
        game.setActionprob()
        import matplotlib.pyplot as plt
        import cv2

        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        from cairosvg import svg2png


        TileShape = {'Circle': 1, 'Square': 2, 'Diamond': 3, 'Clover': 4, 'FourPointStar': 5, 'EightPointStar': 6}
        TileColor = {'Green': 1, 'Blue': 2, 'Purple': 3, 'Red': 4, 'Orange': 5, 'Yellow': 6}
        game.player2.rack = []
        game.player2.rack.append(TileBinaire(3, 1))
        game.player2.rack.append(TileBinaire(3, 1))
        game.player2.rack.append(TileBinaire(5, 1))
        game.player2.rack.append(TileBinaire(1, 6))
        game.player2.rack.append(TileBinaire(3, 6))
        game.player2.rack.append(TileBinaire(3, 4))

        game.isvalid = game.place(TileBinaire(4, 3), 0, 0)
        game.isvalid = game.place(TileBinaire(2, 3), 0, 1)
        game.isvalid = game.place(TileBinaire(4, 5), 1, 0)
        game.isvalid = game.place(TileBinaire(1, 5), 1, -1)
        game.isvalid = game.place(TileBinaire(5, 3), -1, 1)





        from timeit import default_timer as timer
        start = timer()


        game.listValidMovePlayer2()
        end = timer()
        print(end - start)



        assert len(game.listValidMoves)==10

    def test_listvalide_movespermutationNumpy(self):
        game = GameNumpy()
        game.setActionprob()

        TileShape = {'Circle': 1, 'Square': 2, 'Diamond': 3, 'Clover': 4, 'FourPointStar': 5, 'EightPointStar': 6}
        TileColor = {'Green': 1, 'Blue': 2, 'Purple': 3, 'Red': 4, 'Orange': 5, 'Yellow': 6}
        game.player2.zero()
        game.player2.addRack(3, 1)
        game.player2.addRack(3, 1)
        game.player2.addRack(5, 1)
        game.player2.addRack(1, 6)
        game.player2.addRack(3, 6)
        game.player2.addRack(3, 4)
        game.listValidMoves = []
        inp_list = game.player2.getRackList()
        permutations = []

        for i in range(1, len(inp_list) + 1):
            permutations.extend(list(itertools.permutations(inp_list, r=i)))

        if (len(permutations) > 1):
            permutations = np.unique(permutations)
            vfunc = np.vectorize(game.validTilePerùutation)

            permutations = permutations[np.where(vfunc(permutations) == True)]

        assert len(permutations) == 21

    def test_listvalide_movesnot_emptyNumpy(self):
        game = GameNumpy()
        game.setActionprob()


        TileShape = {'Circle': 1, 'Square': 2, 'Diamond': 3, 'Clover': 4, 'FourPointStar': 5, 'EightPointStar': 6}
        TileColor = {'Green': 1, 'Blue': 2, 'Purple': 3, 'Red': 4, 'Orange': 5, 'Yellow': 6}
        game.player2.zero()
        game.player2.addRack(3, 1)
        game.player2.addRack(3, 1)
        game.player2.addRack(5, 1)
        game.player2.addRack(1, 6)
        game.player2.addRack(3, 6)
        game.player2.addRack(3, 4)

        game.isvalid = game.place(4, 3, 0, 0)
        game.isvalid = game.place(2, 3, 0, 1)
        game.isvalid = game.place(4, 5, 1, 0)
        game.isvalid = game.place(1, 5, 1, -1)
        game.isvalid = game.place(5, 3, -1, 1)
        game.isvalid = game.place(5, 1, -2, 1)
        from timeit import default_timer as timer
        start = timer()


        game.listValidMovePlayer2()
        end = timer()
        print(end - start)



        assert len(game.listValidMoves)==10

    def test_placeexempleemptyNumpy(self):
        game = GameNumpy()
        game.setActionprob()


        TileShape = {'Circle': 1, 'Square': 2, 'Diamond': 3, 'Clover': 4, 'FourPointStar': 5, 'EightPointStar': 6}
        TileColor = {'Green': 1, 'Blue': 2, 'Purple': 3, 'Red': 4, 'Orange': 5, 'Yellow': 6}


        game.isvalid = game.place(2, 3, 0, 0)
        game.isvalid = game.place(5, 3, -1, 0)
        game.isvalid = game.place(5, 5, -1, -1)
        game.isvalid = game.place(3, 5, -2, -1)
        game.isvalid = game.place(1, 3, -2, -2)

        import matplotlib.pyplot as plt
        import cv2

        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        from cairosvg import svg2png

        fig, ax = plt.subplots(figsize=(12, 8))

        for x in range(108):
            for y in range(108):
                if game.tilecolor[x, y] != 0:
                    svg2png(url="/home/jcgouleau/PycharmProjects/alphazeroqwirkle/img/" + list(TileColor.keys())[
                        game.tilecolor[x, y]] + list(TileShape.keys())[game.tileshape[x, y]] + ".svg",
                            write_to="stinkbug.png")
                    plt.xlim([0, 50])
                    plt.ylim([0, 50])
                    arr_img = plt.imread("stinkbug.png")
                    half = cv2.resize(arr_img, (0, 0), fx=0.1, fy=0.1)
                    im = OffsetImage(half)

                    ab = AnnotationBbox(im, (25 + (x - 54) * 2.5, 25 + (y - 54) * 3.5), xycoords='data')
                    ax.add_artist(ab)

        plt.show(block=True)
        plt.interactive(False)
        assert game.validBoard() == False

    def test_numberoflisvalidmovefor2tilesinrack(self):
        game = GameBinaire()
        game.setActionprob()



        TileShape = {'Circle': 1, 'Square': 2, 'Diamond': 3, 'Clover': 4, 'FourPointStar': 5, 'EightPointStar': 6}
        TileColor = {'Green': 1, 'Blue': 2, 'Purple': 3, 'Red': 4, 'Orange': 5, 'Yellow': 6}
        game.player2.rack=[]
        game.player2.rack.append(TileBinaire(3,1))
        game.player2.rack.append(TileBinaire(2,1))
        # game.player2.rack.append(TileBinaire(5,1))
        # game.player2.rack.append(TileBinaire(1,6))
        # game.player2.rack.append(TileBinaire(3,6))
        # game.player2.rack.append(TileBinaire(3,4))


        game.listValidMovePlayer2()



        assert len(game.listValidMoves)==2

    def test_numberoflisvalidmovefor2tilesinrackNumpy(self):
        game = GameNumpy()
        game.setActionprob()



        TileShape = {'Circle': 1, 'Square': 2, 'Diamond': 3, 'Clover': 4, 'FourPointStar': 5, 'EightPointStar': 6}
        TileColor = {'Green': 1, 'Blue': 2, 'Purple': 3, 'Red': 4, 'Orange': 5, 'Yellow': 6}

        game.player2.zero()
        game.player2.addRack(3, 1)
        game.player2.addRack(2, 1)


        game.listValidMovePlayer2()



        assert len(game.listValidMoves)==5
    def test_addtilerack(self):
        game = GameNumpy()
        game.setActionprob()



        TileShape = {'Circle': 1, 'Square': 2, 'Diamond': 3, 'Clover': 4, 'FourPointStar': 5, 'EightPointStar': 6}
        TileColor = {'Green': 1, 'Blue': 2, 'Purple': 3, 'Red': 4, 'Orange': 5, 'Yellow': 6}

        game.player1.zero()
        game.player1.addRack(5, 3)
        game.player1.addRack(5, 3)
        game.player1.addRack(2, 6)
        game.player1.addRack(4, 1)
        game.player1.addRack(3, 5)
        game.player1.addRack(6, 1)
        for tile in [[5,3,-1,0]]:
            game.player1.delRack(tile[0],tile[1])






        assert len(np.where(game.player1.getRack()==0))==1

    def test_listvalidmovesneedtonotbenone(self):
        game = GameBinaire()
        game.setActionprob()

        TileShape = {'Circle': 1, 'Square': 2, 'Diamond': 3, 'Clover': 4, 'FourPointStar': 5, 'EightPointStar': 6}
        TileColor = {'Green': 1, 'Blue': 2, 'Purple': 3, 'Red': 4, 'Orange': 5, 'Yellow': 6}
        game.player1.rack = []
        game.player1.rack.append(TileBinaire(2,2))
        game.player1.rack.append(TileBinaire(4, 2))
        game.player1.rack.append(TileBinaire(3,2))
        game.player1.rack.append(TileBinaire(4,4))
        game.player1.rack.append(TileBinaire(6,6))
        game.player1.rack.append(TileBinaire(6,2))
        game.isvalid = game.place(TileBinaire(3, 3), 0, 0)
        game.isvalid = game.place(TileBinaire(2, 3), 0, 1)
        game.isvalid = game.place(TileBinaire(2, 3), 1, 0)
        game.listValidMovePlayer1()
        import matplotlib.pyplot as plt
        import cv2

        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        from cairosvg import svg2png

        fig, ax = plt.subplots(figsize=(12, 8))

        for index, x in enumerate(game.board):
            svg2png(url="/home/jcgouleau/PycharmProjects/alphazeroqwirkle/img/" + x.color + x.shape + ".svg",
                    write_to="stinkbug.png")
            plt.xlim([0, 50])
            plt.ylim([0, 50])
            arr_img = plt.imread("stinkbug.png")
            half = cv2.resize(arr_img, (0, 0), fx=0.1, fy=0.1)
            im = OffsetImage(half)

            ab = AnnotationBbox(im, (25 + x.coordinate.x * 2.5, 25 + x.coordinate.y * 3.5), xycoords='data')
            ax.add_artist(ab)

        plt.show(block=True)
        plt.interactive(False)
        assert game.isvalid==True
        assert len(game.listValidMoves) == 46

    def test_listvalidmovesneedtonotbenoneNumpy(self):
        game = GameNumpy()
        game.setActionprob()

        TileShape = {'Circle': 1, 'Square': 2, 'Diamond': 3, 'Clover': 4, 'FourPointStar': 5, 'EightPointStar': 6}
        TileColor = {'Green': 1, 'Blue': 2, 'Purple': 3, 'Red': 4, 'Orange': 5, 'Yellow': 6}
        game.player1.zero()
        game.player1.addRack(2,2)
        game.player1.addRack(4, 2)
        game.player1.addRack(3, 2)
        game.player1.addRack(4, 4)
        game.player1.addRack(6, 6)
        game.player1.addRack(6, 2)



        game.isvalid = game.place(3, 3, 0, 0)
        game.isvalid = game.place(2, 3, 0, 1)
        game.isvalid = game.place(2, 3, 1, 0)
        game.listValidMovePlayer1()
        import matplotlib.pyplot as plt
        import cv2

        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        from cairosvg import svg2png

        fig, ax = plt.subplots(figsize=(12, 8))

        for x in range(108):
            for y in range(108):
                if game.tilecolor[x, y] != 0:
                    svg2png(url="/home/jcgouleau/PycharmProjects/alphazeroqwirkle/img/" + list(TileColor.keys())[
                        game.tilecolor[x, y]-1] + list(TileShape.keys())[game.tileshape[x, y]-1] + ".svg",
                            write_to="stinkbug.png")
                    plt.xlim([0, 50])
                    plt.ylim([0, 50])
                    arr_img = plt.imread("stinkbug.png")
                    half = cv2.resize(arr_img, (0, 0), fx=0.1, fy=0.1)
                    im = OffsetImage(half)

                    ab = AnnotationBbox(im, (25 + (x - 54) * 2.5, 25 + (y - 54) * 3.5), xycoords='data')
                    ax.add_artist(ab)

        plt.show(block=True)
        plt.interactive(False)
        assert game.isvalid==True
        assert len(game.listValidMoves) == 46

    def test_listvalidmovesneedtonotbenonewith10onboard(self):
        game = GameBinaire()
        game.setActionprob()

        TileShape = {'Circle': 1, 'Square': 2, 'Diamond': 3, 'Clover': 4, 'FourPointStar': 5, 'EightPointStar': 6}
        TileColor = {'Green': 1, 'Blue': 2, 'Purple': 3, 'Red': 4, 'Orange': 5, 'Yellow': 6}
        game.player2.rack = []
        game.player2.rack.append(TileBinaire(4, 4))
        game.player2.rack.append(TileBinaire(2, 2))
        game.player2.rack.append(TileBinaire(6, 3))
        game.player2.rack.append(TileBinaire(4, 2))
        game.player2.rack.append(TileBinaire(1, 4))
        game.player2.rack.append(TileBinaire(3, 6))
        game.isvalid = game.place(TileBinaire(1, 5), 0, 0)
        game.isvalid = game.place(TileBinaire(1, 1), 1, 0)
        game.isvalid = game.place(TileBinaire(5, 1), 1, -1)
        game.isvalid = game.place(TileBinaire(5, 3), 2, -1)
        game.isvalid = game.place(TileBinaire(4, 3), 2, -2)
        game.isvalid = game.place(TileBinaire(6, 3), 3, -2)
        game.isvalid = game.place(TileBinaire(6, 5), 3, -3)
        game.isvalid = game.place(TileBinaire(2, 5), 4, -3)
        game.isvalid = game.place(TileBinaire(3, 5), 5, -3)
        game.isvalid = game.place(TileBinaire(5, 5), 5, -4)
        import datetime
        start_time = datetime.datetime.now().time().strftime('%H:%M:%S.%f')
        game.listValidMovePlayer2()
        end_time = datetime.datetime.now().time().strftime('%H:%M:%S.%f')
        total_time = (datetime.datetime.strptime(end_time, '%H:%M:%S.%f') - datetime.datetime.strptime(start_time,
                                                                                                    '%H:%M:%S.%f'))
        print('total_time:' + str(total_time))
        import matplotlib.pyplot as plt
        import cv2

        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        from cairosvg import svg2png

        fig, ax = plt.subplots(figsize=(12, 8))

        for index, x in enumerate(game.board):
            svg2png(url="/home/jcgouleau/PycharmProjects/alphazeroqwirkle/img/" + x.color + x.shape + ".svg",
                    write_to="stinkbug.png")
            plt.xlim([0, 50])
            plt.ylim([0, 50])
            arr_img = plt.imread("stinkbug.png")
            half = cv2.resize(arr_img, (0, 0), fx=0.1, fy=0.1)
            im = OffsetImage(half)

            ab = AnnotationBbox(im, (25 + x.coordinate.x * 2.5, 25 + x.coordinate.y * 3.5), xycoords='data')
            ax.add_artist(ab)

        for index, x in enumerate(game.player2.rack):
            svg2png(url="/home/jcgouleau/PycharmProjects/alphazeroqwirkle/img/" + list(TileColor.keys())[x.color-1] + list(TileShape.keys())[x.shape-1] + ".svg",
                    write_to="stinkbug.png")
            plt.xlim([0, 50])
            plt.ylim([0, 50])
            arr_img = plt.imread("stinkbug.png")
            half = cv2.resize(arr_img, (0, 0), fx=0.1, fy=0.1)
            im = OffsetImage(half)

            ab = AnnotationBbox(im, (index* 2.5, 0 + x.coordinate.y * 3.5), xycoords='data')
            ax.add_artist(ab)
        plt.show(block=True)
        plt.interactive(False)
        assert game.isvalid == True
        assert len(game.listValidMoves) == 11

    def test_listvalidmovesneedtonotbenonewith10onboardNumpy(self):
        game = GameNumpy()
        game.setActionprob()

        TileShape = {'Circle': 1, 'Square': 2, 'Diamond': 3, 'Clover': 4, 'FourPointStar': 5, 'EightPointStar': 6}
        TileColor = {'Green': 1, 'Blue': 2, 'Purple': 3, 'Red': 4, 'Orange': 5, 'Yellow': 6}

        game.player2.zero()
        game.player2.addRack(4, 4)
        game.player2.addRack(2, 2)
        game.player2.addRack(6, 3)
        game.player2.addRack(4, 2)
        game.player2.addRack(1, 4)
        game.player2.addRack(3, 6)


        game.isvalid = game.place(1, 5, 0, 0)
        game.isvalid = game.place(1, 1, 1, 0)
        game.isvalid = game.place(5, 1, 1, -1)
        game.isvalid = game.place(5, 3, 2, -1)
        game.isvalid = game.place(4, 3, 2, -2)
        game.isvalid = game.place(6, 3, 3, -2)
        game.isvalid = game.place(6, 5, 3, -3)
        game.isvalid = game.place(2, 5, 4, -3)
        game.isvalid = game.place(3, 5, 5, -3)
        game.isvalid = game.place(5, 5, 5, -4)
        import datetime
        start_time = datetime.datetime.now().time().strftime('%H:%M:%S.%f')
        game.listValidMovePlayer2()
        end_time = datetime.datetime.now().time().strftime('%H:%M:%S.%f')
        total_time = (datetime.datetime.strptime(end_time, '%H:%M:%S.%f') - datetime.datetime.strptime(start_time,
                                                                                                    '%H:%M:%S.%f'))
        print('total_time:' + str(total_time))
        import matplotlib.pyplot as plt
        import cv2

        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        from cairosvg import svg2png

        fig, ax = plt.subplots(figsize=(12, 8))
        for x in range(108):
            for y in range(108):
                if game.tilecolor[x, y] != 0:
                    svg2png(url="/home/jcgouleau/PycharmProjects/alphazeroqwirkle/img/" + list(TileColor.keys())[
                        game.tilecolor[x, y] - 1] + list(TileShape.keys())[game.tileshape[x, y] - 1] + ".svg",
                            write_to="stinkbug.png")
                    plt.xlim([0, 50])
                    plt.ylim([0, 50])
                    arr_img = plt.imread("stinkbug.png")
                    half = cv2.resize(arr_img, (0, 0), fx=0.1, fy=0.1)
                    im = OffsetImage(half)

                    ab = AnnotationBbox(im, (25 + (x - 54) * 2.5, 25 + (y - 54) * 3.5), xycoords='data')
                    ax.add_artist(ab)
        for x in range(6):
            svg2png(url="/home/jcgouleau/PycharmProjects/alphazeroqwirkle/img/" + list(TileColor.keys())[game.player2.tilecolor[x]-1] + list(TileShape.keys())[game.player2.tileshape[x]-1] + ".svg",
                    write_to="stinkbug.png")
            plt.xlim([0, 50])
            plt.ylim([0, 50])
            arr_img = plt.imread("stinkbug.png")
            half = cv2.resize(arr_img, (0, 0), fx=0.1, fy=0.1)
            im = OffsetImage(half)

            ab = AnnotationBbox(im, (x* 2.5, 0 + 0 * 3.5), xycoords='data')
            ax.add_artist(ab)
        plt.show(block=True)
        plt.interactive(False)
        assert game.isvalid == True
        assert len(game.listValidMoves) == 11

    def test_shouldvalidboardwhensqwirckle(self):
        game = GameBinaire()
        game.isvalid=game.place(TileBinaire(1, 5), 0, 0)
        game.isvalid=game.place(TileBinaire(1, 2), 1, 0)
        game.isvalid=game.place(TileBinaire(1, 4), 2, 0)
        game.isvalid=game.place(TileBinaire(1, 1), 3, 0)
        game.isvalid=game.place(TileBinaire(1, 3), 4,0)
        game.isvalid=game.place(TileBinaire(1, 6), 5, 0)

        game.listValidMovePlayer2()
        assert game.isvalid == True
    
    def test_shouldvalidboardwhensqwirckleNumpy(self):
        game = GameNumpy()
        game.isvalid=game.place(1, 5, 0, 0)
        game.isvalid=game.place(1, 2, 1, 0)
        game.isvalid=game.place(1, 4, 2, 0)
        game.isvalid=game.place(1, 1, 3, 0)
        game.isvalid=game.place(1, 3, 4,0)
        game.isvalid=game.place(1, 6, 5, 0)

        game.listValidMovePlayer2()
        assert game.isvalid == True

    def test_shouldvalidboardwhensqwirckleNumpytest2(self):
        game = GameNumpy()
        game.isvalid = game.place(2, 1, -1, 0)
        game.isvalid = game.place(2, 4, 0, 0)
        game.isvalid = game.place(2, 3, 1, 0)
        game.isvalid = game.place(2, 5, -2, 0)
        game.isvalid = game.place(2, 2, 2, 0)

        game.listValidMovePlayer2()
        assert game.isvalid == True
    def test_shouldvalidboardwhensqwirckleNumpytest(self):
        # [{'tile': [1, 2, 0, 0]}, {'tile': [4, 2, -1, 0]}, {'tile': [4, 6, -1, -1]}, {'tile': [1, 4, 1, 0]},
        #  {'tile': [6, 4, 1, 1]}, {'tile': [6, 6, 2, 1]}, {'tile': [5, 2, 0, -1]}]
        game = GameNumpy()
        game.isvalid = game.place(1, 2, 0, 0)
        game.isvalid = game.place(4, 2, -1, 0)
        game.isvalid = game.place(4, 6, -1, -1)
        game.isvalid = game.place(1, 4, 1, 0)
        game.isvalid = game.place(6, 4, 1, 1)
        game.isvalid = game.place(6, 6, 2, 1)
        game.isvalid = game.place(5, 2, 0, -1)

        # game.listValidMovePlayer2()
        assert game.isvalid == False
    def test_a_party(self):
        game = GameNumpy()
        game.setActionprob()
        import datetime
        start_time = datetime.datetime.now().time().strftime('%H:%M:%S')
        while not ((game.player1.rackCount()==0  or (game.player2.rackCount()==0) and game.bag.bagCount()==0)) and game.test3round():
            game.listValidMovePlayer1()
            if len(game.listValidMoves) > 0:
                choice=randrange(len(game.listValidMoves))

                for tile in list(game.listValidMoves)[choice]:
                    game.place(tile[0],tile[1],tile[2],tile[3])
                    game.player1.delRack(tile[0],tile[1])
                game.player1.addTileToRack(game.bag)
                game.round = 0
            else:
                game.player1.newRack(game.bag)
                game.round+=1

            game.listValidMovePlayer2()
            if len(game.listValidMoves)>0:
                choice=randrange(len(game.listValidMoves))

                for tile in list(game.listValidMoves)[choice]:
                    game.place(tile[0], tile[1], tile[2], tile[3])
                    game.player2.delRack(tile[0], tile[1])
                game.player2.addTileToRack(game.bag)
                game.round=0
            else:
                game.player2.newRack(game.bag)
                game.round += 1


        import matplotlib.pyplot as plt
        import cv2
        end_time = datetime.datetime.now().time().strftime('%H:%M:%S')
        total_time = (datetime.datetime.strptime(end_time, '%H:%M:%S') - datetime.datetime.strptime(start_time,
                                                                                                    '%H:%M:%S'))
        print('total_time:'+str(total_time))


        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        from cairosvg import svg2png
        tileCount=0
        fig, ax = plt.subplots(figsize=(12, 12))
        plt.xlim([0, 108])
        plt.ylim([0, 108])
        for x in range(108):
            for y in range(108):
                if game.tilecolor[x, y] != 0:
                    svg2png(url="/home/jcgouleau/PycharmProjects/alphazeroqwirkle/img/" + list(TileColor.keys())[
                        game.tilecolor[x, y] - 1] + list(TileShape.keys())[game.tileshape[x, y] - 1] + ".svg",
                            write_to="stinkbug.png")

                    arr_img = plt.imread("stinkbug.png")
                    half = cv2.resize(arr_img, (0, 0), fx=13/107, fy=13/107)
                    im = OffsetImage(half)

                    ab = AnnotationBbox(im, (54 + (x - 54) * 90/13, 54 + (y - 54) * 107/13), xycoords='data')
                    ax.add_artist(ab)
                    tileCount+=1

        plt.show(block=True)
        plt.interactive(False)
        assert game.isvalid == True
        assert game.player2.rackCount()+game.player1.rackCount()+game.bag.bagCount()+tileCount==108

    def testfindindexinactionprob(self):
        game = GameNumpy()
        game.setActionprob()
        game.player2.zero()
        game.player2.addRack(4, 4)
        game.player2.addRack(2, 2)
        game.player2.addRack(6, 3)
        game.player2.addRack(4, 2)
        game.player2.addRack(1, 4)
        game.player2.addRack(3, 6)
        game.listValidMovePlayer2()
        import datetime
        start_time = datetime.datetime.now().time().strftime('%H:%M:%S:%fff')
        listindex=findindexinActionprobnumpy(game)
        end_time = datetime.datetime.now().time().strftime('%H:%M:%S:%fff')
        total_time = (datetime.datetime.strptime(end_time, '%H:%M:%S:%fff') - datetime.datetime.strptime(start_time,
                                                                                                    '%H:%M:%S:%fff'))
        print('total_time:' + str(total_time))

        assert len(np.where(1==listindex)[0])==36





    def testshouldfindallnexxstateinvalimoves(self):

        val=[[[2, 1, 0, 0]], [[3, 5, 0, 0]], [[3, 5, 0, 0], [4, 5, 0, 1]], [[4, 3, 0, 0]], [[4, 3, 0, 0], [4, 4, 0, 1]], [[4, 3, 0, 0], [4, 4, 0, 1], [4, 5, 0, 2]], [[4, 3, 0, 0], [4, 4, 0, 1], [4, 5, 0, 2], [4, 6, 0, 3]], [[4, 3, 0, 0], [4, 4, 0, 1], [4, 6, 0, 2]], [[4, 3, 0, 0], [4, 4, 0, 1], [4, 6, 0, 2], [4, 5, 0, 3]], [[4, 3, 0, 0], [4, 5, 0, 1]], [[4, 3, 0, 0], [4, 5, 0, 1], [4, 4, 0, 2]], [[4, 3, 0, 0], [4, 5, 0, 1], [4, 4, 0, 2], [4, 6, 0, 3]], [[4, 3, 0, 0], [4, 5, 0, 1], [4, 6, 0, 2]], [[4, 3, 0, 0], [4, 5, 0, 1], [4, 6, 0, 2], [4, 4, 0, 3]], [[4, 3, 0, 0], [4, 6, 0, 1]], [[4, 3, 0, 0], [4, 6, 0, 1], [4, 4, 0, 2]], [[4, 3, 0, 0], [4, 6, 0, 1], [4, 4, 0, 2], [4, 5, 0, 3]], [[4, 3, 0, 0], [4, 6, 0, 1], [4, 5, 0, 2]], [[4, 3, 0, 0], [4, 6, 0, 1], [4, 5, 0, 2], [4, 4, 0, 3]], [[4, 4, 0, 0]], [[4, 4, 0, 0], [4, 3, 0, 1]], [[4, 4, 0, 0], [4, 3, 0, 1], [4, 5, 0, 2]], [[4, 4, 0, 0], [4, 3, 0, 1], [4, 5, 0, 2], [4, 6, 0, 3]], [[4, 4, 0, 0], [4, 3, 0, 1], [4, 6, 0, 2]], [[4, 4, 0, 0], [4, 3, 0, 1], [4, 6, 0, 2], [4, 5, 0, 3]], [[4, 4, 0, 0], [4, 5, 0, 1]], [[4, 4, 0, 0], [4, 5, 0, 1], [4, 3, 0, 2]], [[4, 4, 0, 0], [4, 5, 0, 1], [4, 3, 0, 2], [4, 6, 0, 3]], [[4, 4, 0, 0], [4, 5, 0, 1], [4, 6, 0, 2]], [[4, 4, 0, 0], [4, 5, 0, 1], [4, 6, 0, 2], [4, 3, 0, 3]], [[4, 4, 0, 0], [4, 6, 0, 1]], [[4, 4, 0, 0], [4, 6, 0, 1], [4, 3, 0, 2]], [[4, 4, 0, 0], [4, 6, 0, 1], [4, 3, 0, 2], [4, 5, 0, 3]], [[4, 4, 0, 0], [4, 6, 0, 1], [4, 5, 0, 2]], [[4, 4, 0, 0], [4, 6, 0, 1], [4, 5, 0, 2], [4, 3, 0, 3]], [[4, 5, 0, 0]], [[4, 5, 0, 0], [3, 5, 0, 1]], [[4, 5, 0, 0], [4, 3, 0, 1]], [[4, 5, 0, 0], [4, 3, 0, 1], [4, 4, 0, 2]], [[4, 5, 0, 0], [4, 3, 0, 1], [4, 4, 0, 2], [4, 6, 0, 3]], [[4, 5, 0, 0], [4, 3, 0, 1], [4, 6, 0, 2]], [[4, 5, 0, 0], [4, 3, 0, 1], [4, 6, 0, 2], [4, 4, 0, 3]], [[4, 5, 0, 0], [4, 4, 0, 1]], [[4, 5, 0, 0], [4, 4, 0, 1], [4, 3, 0, 2]], [[4, 5, 0, 0], [4, 4, 0, 1], [4, 3, 0, 2], [4, 6, 0, 3]], [[4, 5, 0, 0], [4, 4, 0, 1], [4, 6, 0, 2]], [[4, 5, 0, 0], [4, 4, 0, 1], [4, 6, 0, 2], [4, 3, 0, 3]], [[4, 5, 0, 0], [4, 6, 0, 1]], [[4, 5, 0, 0], [4, 6, 0, 1], [4, 3, 0, 2]], [[4, 5, 0, 0], [4, 6, 0, 1], [4, 3, 0, 2], [4, 4, 0, 3]], [[4, 5, 0, 0], [4, 6, 0, 1], [4, 4, 0, 2]], [[4, 5, 0, 0], [4, 6, 0, 1], [4, 4, 0, 2], [4, 3, 0, 3]], [[4, 6, 0, 0]], [[4, 6, 0, 0], [4, 3, 0, 1]], [[4, 6, 0, 0], [4, 3, 0, 1], [4, 4, 0, 2]], [[4, 6, 0, 0], [4, 3, 0, 1], [4, 4, 0, 2], [4, 5, 0, 3]], [[4, 6, 0, 0], [4, 3, 0, 1], [4, 5, 0, 2]], [[4, 6, 0, 0], [4, 3, 0, 1], [4, 5, 0, 2], [4, 4, 0, 3]], [[4, 6, 0, 0], [4, 4, 0, 1]], [[4, 6, 0, 0], [4, 4, 0, 1], [4, 3, 0, 2]], [[4, 6, 0, 0], [4, 4, 0, 1], [4, 3, 0, 2], [4, 5, 0, 3]], [[4, 6, 0, 0], [4, 4, 0, 1], [4, 5, 0, 2]], [[4, 6, 0, 0], [4, 4, 0, 1], [4, 5, 0, 2], [4, 3, 0, 3]], [[4, 6, 0, 0], [4, 5, 0, 1]], [[4, 6, 0, 0], [4, 5, 0, 1], [4, 3, 0, 2]], [[4, 6, 0, 0], [4, 5, 0, 1], [4, 3, 0, 2], [4, 4, 0, 3]], [[4, 6, 0, 0], [4, 5, 0, 1], [4, 4, 0, 2]], [[4, 6, 0, 0], [4, 5, 0, 1], [4, 4, 0, 2], [4, 3, 0, 3]]]

        nextstate=[[4, 3], [4, 4], [4, 5]]
        for tiles in val:
            if nextstate==[[tile[0],tile[1]] for tile in tiles]:
                setp=tiles
                break

        assert setp==[[4, 3, 0, 0], [4, 4, 0, 1], [4, 5, 0, 2]]

    def testwinner(self):
        game = GameNumpy()
        game.setActionprob()
        game.player2.zero()

        assert game.winner()==-1
        game = GameNumpy()
        game.setActionprob()
        game.player1.zero()

        assert game.winner() == 1


    def testgameboardshow(self):
        import matplotlib.pyplot as plt
        import cv2
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        from cairosvg import svg2png

        test=[[1, 4, 53, 54],[2, 4, 54, 54],[2, 3, 55, 54],[2, 3, 54, 55],[6, 3, 55,55]]

        fig, ax = plt.subplots(figsize=(14, 14))
        plt.xlim([-150, 150])
        plt.ylim([-150, 150])
        for tile in test:
            svg2png(url="/home/jcgouleau/PycharmProjects/alphazeroqwirkle/img/" + list(TileColor.keys())[
                tile[0] - 1] + list(TileShape.keys())[tile[1] - 1] + ".svg",
                    write_to="stinkbug.png")

            arr_img = plt.imread("stinkbug.png")
            half = cv2.resize(arr_img, (0, 0), fx=7 / 85, fy=6/ 85)
            im = OffsetImage(half)

            ab = AnnotationBbox(im, (54 + (tile[2] - 54) * 65 / 6, 54 + (tile[3] - 54) * 65 / 5), xycoords='data')
            ax.add_artist(ab)

        plt.show(block=True)
        plt.interactive(False)

    def test_shouldaddpointtoplayerNumpyforaqwirkle(self):
        game = GameNumpy()
        game.isvalid = game.place(1, 5, 0, 0)
        game.isvalid = game.place(1, 2, 1, 0)
        game.isvalid = game.place(1, 4, 2, 0)
        game.isvalid = game.place(1, 1, 3, 0)
        game.isvalid = game.place(1, 3, 4, 0)
        game.listValidMovePlayer2()
        game.place(1, 6, 5, 0)
        game.player2.point += game.getpoint([[5, 0]])

        assert game.player2.point == 12
    def test_shouldaddpointtoplayerNumpy(self):
        game = GameNumpy()
        game.isvalid=game.place(1, 5, 0, 0)
        game.isvalid=game.place(1, 2, 1, 0)
        game.isvalid=game.place(1, 4, 2, 0)
        game.isvalid=game.place(1, 1, 3, 0)
        game.isvalid=game.place(1, 3, 4,0)
        game.isvalid=game.place(1, 6, 5, 0)

        game.listValidMovePlayer2()
        game.place(1, 3, 0, -1)
        game.player2.point +=game.getpoint([[0,-1]])
        assert game.player2.point == 2

        game.listValidMovePlayer2()
        game.place(4, 6, 5, -1)
        game.place(4, 3, 4, -1)
        game.player2.point += game.getpoint([[5, -1],[4,-1]])
        game.listValidMovePlayer2()
        game.place(1, 4, 1, -1)
        game.place(1, 3, 1, -2)
        game.player2.point += game.getpoint([[1, -1], [ 1, -2]])
        assert game.player2.point == 10

        game.player2.zero()
        game.player2.addTileToRack(game.bag)
        game.listValidMovePlayer2()
        game.place(6, 3, 2, -2)
        game.place(6, 1, 2, -3)
        game.place(6, 2, 2, -4)
        game.place(6, 5, 2, -5)
        game.place(6, 4, 2, -6)
        game.place(6, 6, 2, -7)
        game.player2.point += game.getpoint([[2, -2], [ 2, -3], [2, -4], [2, -5],[2,-6],[2,-7]])
        assert game.player2.point == 23
        game.player2.point=0

        game.place(5, 3, 3, -2)
        game.place(3, 3, 4, -2)
        game.player2.point += game.getpoint([[3, -2], [ 4, -2]])
        assert game.player2.point == 6

        import matplotlib.pyplot as plt
        import cv2
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        from cairosvg import svg2png
        fig, ax = plt.subplots(figsize=(12, 12))
        plt.xlim([-150, 150])
        plt.ylim([-150, 150])
        test = 0
        for x in range(108):
            for y in range(108):
                if game.tilecolor[x, y] != 0:
                    svg2png(url="/home/jcgouleau/PycharmProjects/alphazeroqwirkle/img/" + list(TileColor.keys())[
                        game.tilecolor[x, y] - 1] + list(TileShape.keys())[game.tileshape[x, y] - 1] + ".svg",
                            write_to="stinkbug.png")

                    arr_img = plt.imread("stinkbug.png")
                    half = cv2.resize(arr_img, (0, 0), fx=8.5 / 107, fy=7.5 / 107)
                    im = OffsetImage(half)

                    ab = AnnotationBbox(im, (54 + (x - 54) * 107 / 8.5, 54 + (y - 54) * 107 / 7.5), xycoords='data')
                    ax.add_artist(ab)

        plt.show(block=True)
        plt.interactive(False)



    def test_placetempory(self):
        game = GameNumpy()
        # Test placing a game piece at an empty position within the board bounds
        assert (game.placetempory(1, 1, 53, 53), False)

        # Test placing a game piece at a position that is outside the board bounds
        assert (game.placetempory(1, 1, 54, 108), False)

        # Test placing a game piece at a position that is already occupied
        assert(game.placetempory(1, 1, 53, 53), False)

        # Test placing a game piece with the same color but different shape as a game piece in the same row
        assert(game.placetempory(1, 2, 53, 54), True)

        # Test placing a game piece with the same shape but different color as a game piece in the same row
        assert(game.placetempory(2, 1, 53, 55), True)

        # Test placing a game piece with the same color and shape as a game piece in the same row
        assert(game.placetempory(1, 1, 53, 56), False)

        # Test placing a game piece with the same color but different shape as a game piece in the same column
        assert(game.placetempory(1, 3, 54, 53), True)

        # Test placing a game piece with the same shape but different color as a game piece in the same column
        assert(game.placetempory(2, 1, 55, 53), True)

        # Test placing a game piece with the same color and shape as a game piece in the same column
        assert(game.placetempory(1, 1, 56, 53), False)

    def test_placetempory2(self):
        game = GameNumpy()
        # Test placing a game piece at an empty position within the board bounds
        assert game.placetempory(1, 1, 53, 53) == False

        # Test placing a game piece at a position that is outside the board bounds
        assert game.placetempory(1, 1, 54, 108) == False

        # Test placing a game piece at a position that is already occupied
        assert game.placetempory(1, 1, 53, 53) == False

        # Test placing a game piece with the same color but different shape as a game piece in the same row
        assert game.placetempory(1, 2, 53, 54) == False

        # Test placing a game piece with the same shape but different color as a game piece in the same row
        assert game.placetempory(2, 1, 53, 55) == False

        # Test placing a game piece with the same color and shape as a game piece in the same row
        assert game.placetempory(1, 1, 53, 56) == False

        # Test placing a game piece with the same color but different shape as a game piece in the same column
        assert game.placetempory(1, 3, 54, 53) == False

        # Test placing a game piece with the same shape but different color as a game piece in the same column
        assert game.placetempory(2, 1, 55, 53) == False

        # Test placing a game piece with the same color and shape as a game piece in the same column
        assert game.placetempory(1, 1, 56, 53) == False

        # Test placing a game piece with the same color but different shape as a game piece in an adjacent position
        assert game.placetempory(1, 1, 54, 54) == False

        # Test placing a game piece with the same shape but different color as a game piece in an adjacent position
        assert game.placetempory(1, 1, 55, 55) == False

        # Test placing a game piece with the same color and shape as a game piece in an adjacent position
        assert game.placetempory(1, 1, 56, 56) == False

        # Test placing a game piece with the same color but different shape as

