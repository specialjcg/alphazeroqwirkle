from datetime import time
from random import randrange

import numpy as np

from Bag import Game, Tile, Coordinate
from TileColor import TileColor
from TileShape import TileShape
from qwirckleAlphazero import gridNormToBoardPlay, gridNormtoRack, deepGridCopy
from boardPlayToGridNorm import boardPlayToGridNorm
from convertToBoard import convertToBoard


class TestClassDemoInstance:

    def test_one(self):
        game = Game()
        assert len(game.bag.bag) == 96

    def test_tileOnBoard(self):
        game = Game()
        tile1 = Tile(TileColor['Red'], TileShape['FourPointStar'], Coordinate(0, 0))
        game.tileOnBoard.append(tile1)
        tile1 = Tile(TileColor['Red'], TileShape['FourPointStar'], Coordinate(0, -1))
        game.tileOnBoard.append(tile1)
        assert game.validBoard(game.tileOnBoard) == False

    def test_tileOnBoardTrue(self):
        game = Game()
        tile1 = Tile(TileColor['Red'], TileShape['EightPointStar'], Coordinate(0, 0))
        game.tileOnBoard.append(tile1)
        tile1 = Tile(TileColor['Red'], TileShape['FourPointStar'], Coordinate(0, -1))
        game.tileOnBoard.append(tile1)
        tile1 = Tile(TileColor['Red'], TileShape['Circle'], Coordinate(0, -2))
        game.tileOnBoard.append(tile1)
        assert game.validBoard(game.tileOnBoard) == True
    def test_tileOnBoardFalseSAmeonline(self):
        game = Game()
        tile1 = Tile(TileColor['Red'], TileShape['EightPointStar'], Coordinate(0, 0))
        game.tileOnBoard.append(tile1)
        tile1 = Tile(TileColor['Red'], TileShape['FourPointStar'], Coordinate(0, -1))
        game.tileOnBoard.append(tile1)
        tile1 = Tile(TileColor['Red'], TileShape['EightPointStar'], Coordinate(0, -2))
        game.tileOnBoard.append(tile1)
        assert game.validBoard(game.tileOnBoard) == False


    def test_tileOnBoardNotValid(self):
        game = Game()
        tile1 = Tile(TileColor['Red'], TileShape['FourPointStar'], Coordinate(1, 0))
        game.tileOnBoard.append(tile1)
        tile1 = Tile(TileColor['Red'], TileShape['FourPointStar'], Coordinate(0, -1))
        game.tileOnBoard.append(tile1)
        assert game.validBoard(game.tileOnBoard) == False

    def test_tileOnBoardNotValid2Tilesonsamecolumn(self):
        game = Game()

        tile1 = Tile(TileColor['Red'], TileShape['EightPointStar'], Coordinate(0, 1))
        game.tileOnBoard.append(tile1)
        tile1 = Tile(TileColor['Red'], TileShape['EightPointStar'], Coordinate(0, 0))
        game.tileOnBoard.append(tile1)

        assert game.validBoard(game.tileOnBoard) == False

    def test_tileOnBoardNotValid2Tilessamerow(self):
        game = Game()

        tile1 = Tile(TileColor['Red'], TileShape['EightPointStar'], Coordinate(1, 0))
        game.tileOnBoard.append(tile1)
        tile1 = Tile(TileColor['Red'], TileShape['EightPointStar'], Coordinate(0, 0))
        game.tileOnBoard.append(tile1)

        assert game.validBoard(game.tileOnBoard) == False

    def test_tileOnBoardNotValid4Tilessamerow(self):
        game = Game()

        tile1 = Tile(TileColor['Red'], TileShape['Circle'], Coordinate(0, -1))
        game.tileOnBoard.append(tile1)
        tile1 = Tile(TileColor['Orange'], TileShape['Circle'], Coordinate(1, -1))
        game.tileOnBoard.append(tile1)
        tile1 = Tile(TileColor['Red'], TileShape['Circle'], Coordinate(0, -1))
        game.tileOnBoard.append(tile1)
        tile1 = Tile(TileColor['Green'], TileShape['Circle'], Coordinate(1, -1))
        game.tileOnBoard.append(tile1)
        assert game.validBoard(game.tileOnBoard) == False




    def test_tileOnBoardTogridnorm(self):
        game = Game()
        gridnorme = np.zeros(shape=(26, 54, 54))

        tile1 = Tile(TileColor['Red'], TileShape['EightPointStar'], Coordinate(1, 0))
        game.tileOnBoard.append(tile1)
        tile1 = Tile(TileColor['Red'], TileShape['EightPointStar'], Coordinate(0, 0))
        game.tileOnBoard.append(tile1)
        rack = [['Purple', 'Clover', [-14, -1]]]

        gridnorme = boardPlayToGridNorm(gridnorme, rack, 1)

        assert gridnorme[2][8, 21] == 1.0
        assert gridnorme[3][8, 21] == 0.0
        assert gridnorme[9][8, 21] == 1.0

        board = gridNormToBoardPlay(gridnorme)

        assert board == [['Purple', 'Clover', [-14, -1]]]

    def test_tileOnBoardTogridnormandrack(self):
        game = Game()
        gridnorme = np.zeros(shape=(26, 54, 54))

        tile1 = Tile(TileColor['Red'], TileShape['EightPointStar'], Coordinate(1, 0))
        game.tileOnBoard.append(tile1)
        gridnorme = boardPlayToGridNorm(gridnorme, [tile1.gettype()], 1)
        tile1 = Tile(TileColor['Red'], TileShape['FourPointStar'], Coordinate(0, 0))
        game.tileOnBoard.append(tile1)
        gridnorme = boardPlayToGridNorm(gridnorme, [tile1.gettype()], 1)
        rack = game.player1.getRack()
        gridnorme = convertToBoard(gridnorme, rack)

        assert gridnorme[3][22, 22] == 1.0
        assert gridnorme[3][23, 22] == 1.0
        assert gridnorme[4][23, 22] == 0.0
        assert gridnorme[11][23, 22] == 1.0
        assert gridnorme[10][22, 22] == 1.0
        board = gridNormToBoardPlay(gridnorme)

        assert board == [['Red', 'FourPointStar', [0, 0]], ['Red', 'EightPointStar', [1, 0]]]

        rackgrid = gridNormtoRack(gridnorme)

        assert rackgrid == rack

    def test_tilesetactionprob(self):
        game = Game()
        game.setActionprob()

        assert game.actionprob[236347] == [[0, 2, 2, 0, 0], [0, 2, 2, -1, 0]]
        assert game.actionprob[236431] == [[0, 4, 3, 0, 0], [0, 4, 3, 0, -1]]

    def test_tilechangerack(self):
        game = Game()
        val = game.player1.getRack()

        game.player1.newRack(game.bag)
        newrack = game.player1.getRack()
        vallen = len(game.bag.bag)
        assert val != newrack
        assert vallen == 96

    def test_gridcopy(self):

        gridnorme = np.zeros(shape=(26, 54, 54))
        gridnorme[0][22, 22] = 5
        game = deepGridCopy(gridnorme)
        game[1][22][22] = 5
        assert game[1][22][22] != gridnorme[1][22][22]
        assert game[0][22][22] == 5.0
    def test_gamecopy(self):
        game = Game()
        game.setActionprob()
        val = game.player1.getRack()
        newgame=game.__copy__()
        newgame.player1.newRack(newgame.bag)
        assert newgame.player1.getRack() != game.player1.getRack()





    def test_listvalide_movesnot_empty(self):
        game = Game()
        game.setActionprob()
        valplayer2=[['Purple', 'Circle', [0, 0]], ['Purple', 'Circle', [0, 0]], ['Orange', 'Circle', [0, 0]], ['Green', 'EightPointStar', [0, 0]], ['Purple', 'EightPointStar', [0, 0]], ['Purple', 'Clover', [0, 0]]]
        game.player2.rack=[]
        for i in valplayer2:
            tilr=Tile(i[0],i[1],Coordinate(i[2][0],i[2][1]))
            game.player2.rack.append(tilr)

        valboard=[[['Red', 'Diamond', [0, 0]], ['Blue', 'Diamond', [0, 1]]], [['Red', 'FourPointStar', [1, 0]], ['Green', 'FourPointStar', [1, -1]]], [['Orange', 'Diamond', [-1, 1]]]]


        for lign in valboard:
            for i in lign:
                game.tileOnBoard.append(Tile(i[0],i[1],Coordinate(i[2][0],i[2][1])))



        from timeit import default_timer as timer
        start = timer()

        game.listValidMovePlayer2()
        end = timer()
        print(end - start)


        assert len(game.listValidMoves)==2



    def test_listvalide_movesnot_emptyVERSION17(self):
        game = Game()
        game.setActionprob()
        valplayer2= [['Green', 'Clover', [0, 0]], ['Red', 'Circle', [0, 0]], ['Yellow', 'Circle', [0, 0]], ['Green', 'Clover', [0, 0]],
     ['Blue', 'Clover', [0, 0]], ['Yellow', 'FourPointStar', [0, 0]]]

        game.player2.rack=[]
        for i in valplayer2:
            tilr=Tile(i[0],i[1],Coordinate(i[2][0],i[2][1]))
            game.player2.rack.append(tilr)

        valboard=  [[['Red', 'Circle', [0, 0]], ['Blue', 'Circle', [1, 0]]],
     [['Green', 'Circle', [1, 1]], ['Green', 'Diamond', [2, 1]]], [['Red', 'Diamond', [0, -1]]],
     [['Red', 'Diamond', [2, 2]], ['Yellow', 'Diamond', [3, 2]]], [['Orange', 'Circle', [-1, 0]]],
     [['Orange', 'Diamond', [-1, -1]]], [['Red', 'Diamond', [3, 3]], ['Red', 'EightPointStar', [4, 3]]],
     [['Blue', 'EightPointStar', [4, 4]]], [['Green', 'EightPointStar', [5, 4]]],
     [['Orange', 'FourPointStar', [-1, -2]]],
     [['Green', 'FourPointStar', [-2, -2]], ['Blue', 'FourPointStar', [-2, -3]], ['Purple', 'FourPointStar', [-2, -4]]],
     [['Yellow', 'FourPointStar', [-3, -3]]],
     [['Purple', 'EightPointStar', [4, 5]], ['Green', 'EightPointStar', [4, 6]]],
     [['Yellow', 'EightPointStar', [5, 5]]], [['Red', 'FourPointStar', [0, -2]]], [['Blue', 'EightPointStar', [5, 6]]],
     [['Purple', 'EightPointStar', [6, 4]]],[['Yellow', 'FourPointStar', [-1, -4]]]]

        for lign in valboard:
            for i in lign:
                game.tileOnBoard.append(Tile(i[0],i[1],Coordinate(i[2][0],i[2][1])))
        game.listValidMovePlayer2()
        import matplotlib.pyplot as plt
        import cv2

        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        from cairosvg import svg2png

        fig, ax = plt.subplots(figsize=(12, 8))

        for x in valboard:
            for tile in x:
                svg2png(url="/home/jcgouleau/PycharmProjects/alphazeroqwirkle/img/" + tile[0] + tile[1] + ".svg",
                        write_to="stinkbug.png")
                plt.xlim([0, 50])
                plt.ylim([0, 50])
                arr_img = plt.imread("stinkbug.png")
                half = cv2.resize(arr_img, (0, 0), fx=0.1, fy=0.1)
                im = OffsetImage(half)

                ab = AnnotationBbox(im, (25 + tile[2][0] * 2.5, 25 + tile[2][1] * 3.5), xycoords='data')
                ax.add_artist(ab)

        plt.show(block=True)
        plt.interactive(False)

        assert game.validBoard(game.tileOnBoard) == True
        assert len(game.listValidMoves)==4

    def test_listvalide_movesnot_emptyVERSIONTWO(self):
        game = Game()
        game.setActionprob()
        valplayer2=[['Blue', 'Square', [0, 0]], ['Blue', 'Clover', [0, 0]], ['Green', 'FourPointStar', [0, 0]],
     ['Purple', 'Diamond', [0, 0]], ['Purple', 'FourPointStar', [0, 0]], ['Orange', 'Clover', [0, 0]]]
        game.player2.rack=[]
        for i in valplayer2:
            tilr=Tile(i[0],i[1],Coordinate(i[2][0],i[2][1]))
            game.player2.rack.append(tilr)

        valboard= [[['Purple', 'Diamond', [0, 0]], ['Purple', 'EightPointStar', [0, -1]]], [['Green', 'EightPointStar', [-1, -1]]],
     [['Green', 'Diamond', [-1, -2]], ['Green', 'Clover', [-1, -3]], ['Green', 'FourPointStar', [-1, -4]]],
     [['Red', 'Diamond', [-2, -2]]], [['Red', 'EightPointStar', [-2, -1]], ['Orange', 'EightPointStar', [-3, -1]]]]

        for lign in valboard:
            for i in lign:
                game.tileOnBoard.append(Tile(i[0],i[1],Coordinate(i[2][0],i[2][1])))
        game.listValidMovePlayer2()
        import matplotlib.pyplot as plt
        import cv2

        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        from cairosvg import svg2png

        fig, ax = plt.subplots(figsize=(12, 8))

        for x in valboard:
            for tile in x:
                svg2png(url="/home/jcgouleau/PycharmProjects/alphazeroqwirkle/img/" + tile[0] + tile[1] + ".svg",
                        write_to="stinkbug.png")
                plt.xlim([0, 50])
                plt.ylim([0, 50])
                arr_img = plt.imread("stinkbug.png")
                half = cv2.resize(arr_img, (0, 0), fx=0.1, fy=0.1)
                im = OffsetImage(half)

                ab = AnnotationBbox(im, (25 + tile[2][0] * 2.5, 25 + tile[2][1] * 3.5), xycoords='data')
                ax.add_artist(ab)

        plt.show(block=True)
        plt.interactive(False)


        assert len(game.listValidMoves)==10


    def test_allvalidMove(self):
        game = Game()
        game.player1.rack=[]

        valplayer2=[['Orange', 'EightPointStar', [0, 0]], ['Blue', 'Square', [0, 0]], ['Red', 'FourPointStar', [0, 0]],
         ['Red', 'FourPointStar', [0, 0]], ['Green', 'Square', [0, 0]], ['Purple', 'Circle', [0, 0]]]
        for i in valplayer2:
            tilr=Tile(i[0],i[1],Coordinate(i[2][0],i[2][1]))
            game.player1.rack.append(tilr)
        boardplay=[[['Green', 'Square', [-1, 0]],['Green', 'Diamond', [0, 0]], ['Green', 'EightPointStar', [1, 0]]],[['Green', 'EightPointStar', [0, 1]], ['Green', 'Clover', [1, 1]]]]
        for lign in boardplay:
            for i in lign:
                game.tileOnBoard.append(Tile(i[0],i[1],Coordinate(i[2][0],i[2][1])))


        import matplotlib.pyplot as plt
        import cv2

        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        from cairosvg import svg2png



        fig, ax = plt.subplots(figsize=(12, 8))


        for x in boardplay:
            for tile in x:
                svg2png(
                    url="/home/jcgouleau/PycharmProjects/alphazeroqwirkle/img/" + tile[0] + tile[1] + ".svg",
                    write_to="stinkbug.png")
                plt.xlim([0, 50])
                plt.ylim([0, 50])
                arr_img = plt.imread("stinkbug.png")
                half = cv2.resize(arr_img, (0, 0), fx=0.1, fy=0.1)
                im = OffsetImage(half)

                ab = AnnotationBbox(im, (25 + tile[2][0] * 2.5, 25 + tile[2][1] * 3.5), xycoords='data')
                ax.add_artist(ab)

        plt.show(block=True)
        plt.interactive(False)


        game.listValidMovePlayer1()


        assert game.validBoard(game.tileOnBoard) == True


    def test_shouldproposallistvalidmoveforgame2(self):
        game = Game()
        game.player2.rack=[]

        valplayer2=  [['Yellow', 'Diamond', [0, 0]], ['Yellow', 'Diamond', [0, 0]], ['Blue', 'Clover', [0, 0]],
     ['Yellow', 'Diamond', [0, 0]], ['Blue', 'EightPointStar', [0, 0]], ['Red', 'Diamond', [0, 0]]]
        for i in valplayer2:
            tilr=Tile(i[0],i[1],Coordinate(i[2][0],i[2][1]))
            game.player2.rack.append(tilr)
        boardplay=[[['Purple', 'Circle', [0, 0]], ['Green', 'Circle', [1, 0]], ['Orange', 'Circle', [2, 0]]],
     [['Green', 'FourPointStar', [1, 1]]], [['Orange', 'FourPointStar', [2, 1]], ['Red', 'FourPointStar', [3, 1]]],
     [['Green', 'EightPointStar', [1, 2]]], [['Blue', 'EightPointStar', [0, 2]], ['Blue', 'Square', [0, 3]]]]
        for lign in boardplay:
            for i in lign:
                game.tileOnBoard.append(Tile(i[0],i[1],Coordinate(i[2][0],i[2][1])))


        import matplotlib.pyplot as plt
        import cv2

        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        from cairosvg import svg2png



        fig, ax = plt.subplots(figsize=(12, 8))


        for x in boardplay:
            for tile in x:
                svg2png(
                    url="/home/jcgouleau/PycharmProjects/alphazeroqwirkle/img/" + tile[0] + tile[1] + ".svg",
                    write_to="stinkbug.png")
                plt.xlim([0, 50])
                plt.ylim([0, 50])
                arr_img = plt.imread("stinkbug.png")
                half = cv2.resize(arr_img, (0, 0), fx=0.1, fy=0.1)
                im = OffsetImage(half)

                ab = AnnotationBbox(im, (25 + tile[2][0] * 2.5, 25 + tile[2][1] * 3.5), xycoords='data')
                ax.add_artist(ab)

        plt.show(block=True)
        plt.interactive(False)


        game.listValidMovePlayer2()


        assert game.validBoard(game.tileOnBoard) == True

    def test_shouldnottrueforvalidMove_forgreeneightpoinstaronsameproposal(self):
        game = Game()
        game.player1.rack=[]

        valplayer2=[['Orange', 'EightPointStar', [0, 0]], ['Blue', 'Square', [0, 0]], ['Red', 'FourPointStar', [0, 0]],
         ['Red', 'FourPointStar', [0, 0]], ['Green', 'Square', [0, 0]], ['Purple', 'Circle', [0, 0]]]
        for i in valplayer2:
            tilr=Tile(i[0],i[1],Coordinate(i[2][0],i[2][1]))
            game.player1.rack.append(tilr)
        boardplay=[[['Green', 'EightPointStar', [-1, 0]],['Green', 'Diamond', [0, 0]], ['Green', 'EightPointStar', [1, 0]],['Green', 'EightPointStar', [0, 1]], ['Green', 'Clover', [1, 1]]]]
        for lign in boardplay:
            for i in lign:
                game.tileOnBoard.append(Tile(i[0],i[1],Coordinate(i[2][0],i[2][1])))


        import matplotlib.pyplot as plt
        import cv2

        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        from cairosvg import svg2png



        fig, ax = plt.subplots(figsize=(12, 8))


        for x in boardplay:
            for tile in x:
                svg2png(
                    url="/home/jcgouleau/PycharmProjects/alphazeroqwirkle/img/" + tile[0] + tile[1] + ".svg",
                    write_to="stinkbug.png")
                plt.xlim([0, 50])
                plt.ylim([0, 50])
                arr_img = plt.imread("stinkbug.png")
                half = cv2.resize(arr_img, (0, 0), fx=0.1, fy=0.1)
                im = OffsetImage(half)

                ab = AnnotationBbox(im, (25 + tile[2][0] * 2.5, 25 + tile[2][1] * 3.5), xycoords='data')
                ax.add_artist(ab)

        plt.show(block=True)
        plt.interactive(False)


        game.listValidMovePlayer1()


        assert game.validBoard(game.tileOnBoard) == False






    def test_shouldnotvalidMove_forgreeneightpoinstaronsameproposal(self):
        game = Game()
        game.player1.rack=[]

        valplayer2=[['Orange', 'EightPointStar', [0, 0]], ['Blue', 'Square', [0, 0]], ['Red', 'FourPointStar', [0, 0]],
         ['Red', 'FourPointStar', [0, 0]], ['Green', 'Square', [0, 0]], ['Purple', 'Circle', [0, 0]]]
        for i in valplayer2:
            tilr=Tile(i[0],i[1],Coordinate(i[2][0],i[2][1]))
            game.player1.rack.append(tilr)
        boardplay=[[['Red', 'FourPointStar', [0, 0]]], [['Blue', 'FourPointStar', [-1, 0]]], [['Red', 'FourPointStar', [-1, 1]]], [['Green', 'Square', [0, -1]]]]
        for lign in boardplay:
            for i in lign:
                game.tileOnBoard.append(Tile(i[0],i[1],Coordinate(i[2][0],i[2][1])))


        # import matplotlib.pyplot as plt
        # import cv2
        #
        # from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        # from cairosvg import svg2png
        #
        #
        #
        # fig, ax = plt.subplots(figsize=(12, 8))
        #
        #
        # for x in boardplay:
        #     for tile in x:
        #         svg2png(
        #             url="/home/jcgouleau/PycharmProjects/alphazeroqwirkle/img/" + tile[0] + tile[1] + ".svg",
        #             write_to="stinkbug.png")
        #         plt.xlim([0, 50])
        #         plt.ylim([0, 50])
        #         arr_img = plt.imread("stinkbug.png")
        #         half = cv2.resize(arr_img, (0, 0), fx=0.1, fy=0.1)
        #         im = OffsetImage(half)
        #
        #         ab = AnnotationBbox(im, (25 + tile[2][0] * 2.5, 25 + tile[2][1] * 3.5), xycoords='data')
        #         ax.add_artist(ab)
        #
        # plt.show(block=True)
        # plt.interactive(False)


        game.listValidMovePlayer1()


        assert game.validBoard(game.tileOnBoard) == False


    def test_shouldnotvalidMove_for_PurpleClover05(self):
        game = Game()
        game.player1.rack=[]

        valplayer2=[['Orange', 'EightPointStar', [0, 0]], ['Blue', 'Square', [0, 0]], ['Red', 'FourPointStar', [0, 0]],
         ['Red', 'FourPointStar', [0, 0]], ['Green', 'Square', [0, 0]], ['Purple', 'Circle', [0, 0]]]
        for i in valplayer2:
            tilr=Tile(i[0],i[1],Coordinate(i[2][0],i[2][1]))
            game.player1.rack.append(tilr)
        boardplay= [[['Purple', 'EightPointStar', [0, 0]], ['Red', 'EightPointStar', [0, 1]], ['Yellow', 'EightPointStar', [0, 2]]], [['Red', 'Clover', [1, 1]]], [['Green', 'Diamond', [2, -1]]]]

        for lign in boardplay:
            for i in lign:
                game.tileOnBoard.append(Tile(i[0],i[1],Coordinate(i[2][0],i[2][1])))


        # import matplotlib.pyplot as plt
        # import cv2
        #
        # from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        # from cairosvg import svg2png
        #
        #
        #
        # fig, ax = plt.subplots(figsize=(12, 8))
        #
        #
        # for x in boardplay:
        #     for tile in x:
        #         svg2png(
        #             url="/home/jcgouleau/PycharmProjects/alphazeroqwirkle/img/" + tile[0] + tile[1] + ".svg",
        #             write_to="stinkbug.png")
        #         plt.xlim([0, 50])
        #         plt.ylim([0, 50])
        #         arr_img = plt.imread("stinkbug.png")
        #         half = cv2.resize(arr_img, (0, 0), fx=0.1, fy=0.1)
        #         im = OffsetImage(half)
        #
        #         ab = AnnotationBbox(im, (25 + tile[2][0] * 2.5, 25 + tile[2][1] * 3.5), xycoords='data')
        #         ax.add_artist(ab)
        #
        # plt.show(block=True)
        # plt.interactive(False)


        game.listValidMovePlayer1()


        assert game.validBoard(game.tileOnBoard) == False


    def test_shouldnotvalidMove_for_OrangeFourPointStar(self):
        game = Game()
        game.player1.rack=[]

        valplayer2=[['Orange', 'EightPointStar', [0, 0]], ['Blue', 'Square', [0, 0]], ['Red', 'FourPointStar', [0, 0]],
         ['Red', 'FourPointStar', [0, 0]], ['Green', 'Square', [0, 0]], ['Purple', 'Circle', [0, 0]]]
        for i in valplayer2:
            tilr=Tile(i[0],i[1],Coordinate(i[2][0],i[2][1]))
            game.player1.rack.append(tilr)
        boardplay=   [[['Orange', 'FourPointStar', [0, 0]], ['Orange', 'EightPointStar', [0, 1]]], [['Orange', 'FourPointStar', [1, -1]], ['Orange', 'EightPointStar', [1, -2]], ['Orange', 'Clover', [1, -3]]], [['Orange', 'Clover', [-2, 1]]], [['Red', 'FourPointStar', [2, 0]]]]

        for lign in boardplay:
            for i in lign:
                game.tileOnBoard.append(Tile(i[0],i[1],Coordinate(i[2][0],i[2][1])))


        # import matplotlib.pyplot as plt
        # import cv2
        #
        # from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        # from cairosvg import svg2png
        #
        #
        #
        # fig, ax = plt.subplots(figsize=(12, 8))
        #
        #
        # for x in boardplay:
        #     for tile in x:
        #         svg2png(
        #             url="/home/jcgouleau/PycharmProjects/alphazeroqwirkle/img/" + tile[0] + tile[1] + ".svg",
        #             write_to="stinkbug.png")
        #         plt.xlim([0, 50])
        #         plt.ylim([0, 50])
        #         arr_img = plt.imread("stinkbug.png")
        #         half = cv2.resize(arr_img, (0, 0), fx=0.1, fy=0.1)
        #         im = OffsetImage(half)
        #
        #         ab = AnnotationBbox(im, (25 + tile[2][0] * 2.5, 25 + tile[2][1] * 3.5), xycoords='data')
        #         ax.add_artist(ab)
        #
        # plt.show(block=True)
        # plt.interactive(False)


        game.listValidMovePlayer1()


        assert game.validBoard(game.tileOnBoard) == False



    def test_shouldnotvalidMove_for_GreenFourPointStarPurpleFourPointStar(self):
        game = Game()
        game.player1.rack = []

        valplayer2 = [['Orange', 'EightPointStar', [0, 0]], ['Blue', 'Square', [0, 0]],
                      ['Red', 'FourPointStar', [0, 0]],
                      ['Red', 'FourPointStar', [0, 0]], ['Green', 'Square', [0, 0]], ['Purple', 'Circle', [0, 0]]]
        for i in valplayer2:
            tilr = Tile(i[0], i[1], Coordinate(i[2][0], i[2][1]))
            game.player1.rack.append(tilr)
        boardplay = [[['Green', 'FourPointStar', [0, 0]], ['Purple', 'FourPointStar', [0, 1]]],
     [['Yellow', 'FourPointStar', [1, 1]], ['Yellow', 'Square', [1, 2]]], [['Yellow', 'Clover', [2, 1]]]]

        for lign in boardplay:
            for i in lign:
                game.tileOnBoard.append(Tile(i[0], i[1], Coordinate(i[2][0], i[2][1])))

        import matplotlib.pyplot as plt
        import cv2

        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        from cairosvg import svg2png

        fig, ax = plt.subplots(figsize=(12, 8))

        for x in boardplay:
            for tile in x:
                svg2png(
                    url="/home/jcgouleau/PycharmProjects/alphazeroqwirkle/img/" + tile[0] + tile[1] + ".svg",
                    write_to="stinkbug.png")
                plt.xlim([0, 50])
                plt.ylim([0, 50])
                arr_img = plt.imread("stinkbug.png")
                half = cv2.resize(arr_img, (0, 0), fx=0.1, fy=0.1)
                im = OffsetImage(half)

                ab = AnnotationBbox(im, (25 + tile[2][0] * 2.5, 25 + tile[2][1] * 3.5), xycoords='data')
                ax.add_artist(ab)

        plt.show(block=True)
        plt.interactive(False)

        game.listValidMovePlayer1()

        assert game.validBoard(game.tileOnBoard) == False

    def test_shouldnotvalidboardwhensameshapeandcoloronlineorcolumnconsecutive(self):
        game = Game()

        boardplay = [[['Green', 'FourPointStar', [0, 0]],['Green', 'Square', [1, 0]],['Green', 'Clover', [2, 0]],
                      ['Green', 'Circle', [3, 0]],['Green', 'Diamond', [4, 0]],['Green', 'EightPointStar', [5, 0]],['Green', 'FourPointStar', [6,0]]]]

        for lign in boardplay:
            for i in lign:
                game.tileOnBoard.append(Tile(i[0], i[1], Coordinate(i[2][0], i[2][1])))



        assert game.validBoard(game.tileOnBoard) == False

    def test_shouldnotvalidboardsameposition(self):
        game = Game()

        boardplay = [[['Green', 'FourPointStar', [0, 0]], ['Green', 'Square', [1, 0]], ['Green', 'Clover', [2, 0]],
                      ['Green', 'Circle', [3, 0]], ['Green', 'Diamond', [0, 0]], ['Green', 'EightPointStar', [5, 0]]]]
        for lign in boardplay:
            for i in lign:
                game.tileOnBoard.append(Tile(i[0], i[1], Coordinate(i[2][0], i[2][1])))



        assert game.goodPositionTempory(game.tileOnBoard) == False

    def test_shouldvalidboardwhensqwirckle(self):
        game = Game()

        boardplay = [[['Green', 'FourPointStar', [0, 0]], ['Green', 'Square', [1, 0]], ['Green', 'Clover', [2, 0]],
                      ['Green', 'Circle', [3, 0]], ['Green', 'Diamond', [4, 0]], ['Green', 'EightPointStar', [5, 0]]]]

        for lign in boardplay:
            for i in lign:
                game.tileOnBoard.append(Tile(i[0], i[1], Coordinate(i[2][0], i[2][1])))

        assert game.validBoard(game.tileOnBoard) == True

    def test_shouldnotvalidMove_for_twoFourPointStarOnsamelign(self):
        game = Game()


        boardplay = [[['Green', 'FourPointStar', [0, 0]]], [['Blue', 'FourPointStar', [-1, 0]]],
     [['Orange', 'FourPointStar', [-1, -1]]],
     [['Purple', 'FourPointStar', [0, -2]], ['Purple', 'FourPointStar', [-1, -2]]],
     [['Purple', 'EightPointStar', [-2, -2]]]]

        for lign in boardplay:
            for i in lign:
                game.tileOnBoard.append(Tile(i[0], i[1], Coordinate(i[2][0], i[2][1])))

        import matplotlib.pyplot as plt
        import cv2

        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        from cairosvg import svg2png

        fig, ax = plt.subplots(figsize=(12, 8))

        for x in boardplay:
            for tile in x:
                svg2png(
                    url="/home/jcgouleau/PycharmProjects/alphazeroqwirkle/img/" + tile[0] + tile[1] + ".svg",
                    write_to="stinkbug.png")
                plt.xlim([0, 108])
                plt.ylim([0,108])
                arr_img = plt.imread("stinkbug.png")
                half = cv2.resize(arr_img, (0, 0), fx=0.05, fy=0.05)
                im = OffsetImage(half)

                ab = AnnotationBbox(im, (54 + tile[2][0] * 4.5, 54 + tile[2][1] * 6.5), xycoords='data')
                ax.add_artist(ab)

        plt.show(block=True)
        plt.interactive(False)

        # game.listValidMovePlayer1()

        assert game.validBoard(game.tileOnBoard) == False
    def test_shouldnotvalidMove_for_twoFourPointStarOnsamelign(self):
        game = Game()
        # TileColor = {'Green': 1, 'Blue': 2, 'Purple': 3, 'Red': 4, 'Orange': 5, 'Yellow': 6}
        # TileShape = {'Circle': 1, 'Square': 2, 'Diamond': 3, 'Clover': 4, 'FourPointStar': 5, 'EightPointStar': 6}
        # [{'tile': [1, 2, 0, 0]}, {'tile': [4, 2, -1, 0]}, {'tile': [4, 6, -1, -1]}, {'tile': [1, 4, 1, 0]},
        #  {'tile': [6, 4, 1, 1]}, {'tile': [6, 6, 2, 1]}, {'tile': [5, 2, 0, -1]}]
        boardplay = [[['Green', 'Square', [0, 0]]], [['Red', 'Square', [-1, 0]]],
     [['Red', 'EightPointStar', [-1, -1]]],
     [['Green', 'Clover', [1, 0]], ['Yellow', 'Clover', [1, 1]]],
     [['Yellow', 'EightPointStar', [2, 1]],['Orange', 'Square', [0, -1]]]]

        for lign in boardplay:
            for i in lign:
                game.tileOnBoard.append(Tile(i[0], i[1], Coordinate(i[2][0], i[2][1])))

        import matplotlib.pyplot as plt
        import cv2

        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        from cairosvg import svg2png

        fig, ax = plt.subplots(figsize=(12, 8))

        for x in boardplay:
            for tile in x:
                svg2png(
                    url="/home/jcgouleau/PycharmProjects/alphazeroqwirkle/img/" + tile[0] + tile[1] + ".svg",
                    write_to="stinkbug.png")
                plt.xlim([0, 108])
                plt.ylim([0,108])
                arr_img = plt.imread("stinkbug.png")
                half = cv2.resize(arr_img, (0, 0), fx=0.05, fy=0.05)
                im = OffsetImage(half)

                ab = AnnotationBbox(im, (54 + tile[2][0] * 4.5, 54 + tile[2][1] * 6.5), xycoords='data')
                ax.add_artist(ab)

        plt.show(block=True)
        plt.interactive(False)

        # game.listValidMovePlayer1()

        assert game.validBoard(game.tileOnBoard) == False


    def test_a_party(self):
        game = Game()
        game.setActionprob()
        boardPlay=[]
        import time
        import datetime
        start_time = datetime.datetime.now().time().strftime('%H:%M:%S')
        while len(game.player1.rack)!=0 and len(game.player2.rack)!=0:
            game.listValidMovePlayer1()
            if len(game.listValidMoves) > 0:
                alllistvalidmoves = [[j.get() for j in i] for i in game.listValidMoves]
                choice=randrange(len(game.listValidMoves))

                for tile in alllistvalidmoves[choice]:
                    tile = Tile(tile[0], tile[1], Coordinate(tile[2][0], tile[2][1]))
                    game.tileOnBoard.append(tile)
                    game.player1.delRack(tile)
            else:
                for tile in game.player1.rack:
                    game.bag.bag.append(tile)
                game.player1.rack = []
            game.player1.addTileToRack(game.bag)
            game.listValidMovePlayer2()
            if len(game.listValidMoves)>0:
                alllistvalidmoves = [[j.get() for j in i] for i in game.listValidMoves]
                choice=randrange(len(game.listValidMoves))

                for tile in alllistvalidmoves[choice]:
                    tile = Tile(tile[0], tile[1], Coordinate(tile[2][0], tile[2][1]))
                    game.tileOnBoard.append(tile)
                    game.player2.delRack(tile)
            else:
                for tile in game.player2.rack:
                    game.bag.bag.append(tile)
                game.player2.rack=[]

            game.player2.addTileToRack(game.bag)
        import matplotlib.pyplot as plt
        import cv2
        end_time = datetime.datetime.now().time().strftime('%H:%M:%S')
        total_time = (datetime.datetime.strptime(end_time, '%H:%M:%S') - datetime.datetime.strptime(start_time,
                                                                                                    '%H:%M:%S'))
        print('total_time:'+str(total_time))
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        from cairosvg import svg2png

        fig, ax = plt.subplots(figsize=(12, 8))
        for tile in game.tileOnBoard:
                svg2png(url="/home/jcgouleau/PycharmProjects/alphazeroqwirkle/img/" + tile.color + tile.shape + ".svg",
                        write_to="stinkbug.png")
                plt.xlim([0, 216])
                plt.ylim([0, 216])
                arr_img = plt.imread("stinkbug.png")
                half = cv2.resize(arr_img, (0, 0), fx=0.08, fy=0.08)
                im = OffsetImage(half)

                ab = AnnotationBbox(im, (108 + tile.coordinate.x * 8.5, 108 + tile.coordinate.y * 11.5), xycoords='data')
                ax.add_artist(ab)

        plt.show(block=True)
        plt.interactive(False)
        assert game.validBoard(game.tileOnBoard) == True