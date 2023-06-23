import cv2
from cairosvg import svg2png
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from GameNumpy import GameNumpy
from TileColor import TileColor
from TileShape import TileShape


def test_shouldaddpointtoplayerNumpy():
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
    assert game.player2.point == 8

    game.listValidMovePlayer2()
    game.place(1, 4, 1, -1)
    game.place(1, 3, 1, -2)
    game.player2.point += game.getpoint([[1, -1], [ 1, -2]])
    assert game.player2.point == 13



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
    assert game.player2.point == 27

    game.place(5, 3, 3, -2)
    game.place(3, 3, 4, -2)

    game.player2.point += game.getpoint([[3, -2], [ 4, -2]])
    assert game.player2.point == 34
    game.isvalid = game.place(1, 4,0, 1)
    game.isvalid = game.place(1, 5,-1, 1)
    game.isvalid = game.place(1, 3,-2, 1)



    game.player2.point += game.getpoint([[0, 1], [-1, 1], [-2, 1]])
    assert game.player2.point == 40

    game.isvalid = game.place(1, 2, -3, 1)

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
    game.player2.point += game.getpoint([[-3, 1]])
    assert game.player2.point == 44