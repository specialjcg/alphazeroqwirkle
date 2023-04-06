import multiprocessing
import pickle
import datetime

import numpy as np
import torch
from flask import Flask, request
from flask_cors import CORS, cross_origin

import GameNumpy as newGame
from qwirckleAlphazero import convertToBoard, convertToRealTiles, mcts, loadbrain1

app = Flask(__name__)
CORS(app)
game = newGame.GameNumpy()
        #game.setActionprobtest()
game.actionprob = pickle.load(open('gameActionProb.pkl', 'rb'))
gridnorme = np.zeros(shape=(26,54, 54))
import json

loadbrain1()
import jsonpickle
tileonboard = []



@app.route("/play", methods=['GET'])
@cross_origin()
def play():
    global gridnorme, tileonboard,game
    if request.method == 'GET':
        game.listValidMovePlayer1All()
        if len(game.listValidMoves) > 0:
            gridnorme = convertToBoard(gridnorme, game.player1.getRack())
            to_play=1
            action=0
            numsimul=400
            start_time = datetime.datetime.now().time().strftime('%H:%M:%S')
            with multiprocessing.Pool(processes=8) as pool:
                results = pool.starmap(
                    mcts.run,
                    [( gridnorme.copy(), to_play, action, game.__copy__(), numsimul,i) for i in range(5)]
                )

            # get the results from the multiprocessing pool
            actions = sorted(results, key=lambda x: x.prior, reverse=True)[0:1][0]

            # close the multiprocessing pool
            pool.close()
            # actions=mcts.run(gridnorme,1,0,game.__copy__(),50)


            # actions = torch.tensor([actions.children[visit].visit_count for visit in actions.children],
            #                        dtype=torch.float32).cuda()
            childvisitcount = torch.tensor(
                [[actions.children[visit].visit_count] for visit in actions.children],
                dtype=torch.float32)
            childvisitcount /= torch.sum(childvisitcount)
            # game.setActionprob()
            actionPosition = list(actions.children.keys())[
                torch.sort(childvisitcount, descending=True).indices[0]]
            boardPlay = game.actionprob[actionPosition]

            boardPlay = convertToRealTiles(boardPlay,game)
            if boardPlay in game.listValidMoves:
                for tile in boardPlay:
                    if game.place(tile[0], tile[1], tile[2], tile[3]) :
                        tileonboard.append({'tile':[int(tile[0]),int(tile[1]),int(tile[2]), int(tile[3])]})
                        game.player1.delRack(tile[0], tile[1])
                    else:
                        game.round += 1
                        break


                game.player1.addTileToRack(game.bag)

                game.round = 0


                game.player1.point += game.getpoint([[x[2], x[3]] for x in boardPlay])
            else:
                game.player1.newRack(game.bag)
                game.round += 1


        else:
            game.player1.newRack(game.bag)
            game.round += 1

    end_time = datetime.datetime.now().time().strftime('%H:%M:%S')
    total_time = (datetime.datetime.strptime(end_time, '%H:%M:%S') - datetime.datetime.strptime(start_time,
                                                                                                '%H:%M:%S'))
    print('total_time:' + str(total_time))
    empJSON = jsonpickle.encode(tileonboard, unpicklable=False)
    return empJSON
@app.route("/player2", methods=['GET'])
@cross_origin()
def player2():
    global gridnorme, tileonboard,game
    if request.method == 'GET':
        player_dict = {
            "point": game.player2.point,
            "tilecolor": game.player2.tilecolor.tolist(),
            "tileshape": game.player2.tileshape.tolist()
        }


        json_string = json.dumps(player_dict)

        return json_string
@app.route("/player1", methods=['GET'])
@cross_origin()
def player1():
    global gridnorme, tileonboard,game
    if request.method == 'GET':
        player_dict = {
            "point": game.player1.point,
            "tilecolor": game.player1.tilecolor.tolist(),
            "tileshape": game.player1.tileshape.tolist()
        }


        json_string = json.dumps(player_dict)

        return json_string
@app.route("/player2play", methods=['POST'])
@cross_origin()
def player2play():
    global gridnorme, tileonboard,game
    if request.method == 'POST':
        request_data = request.get_json()
        tiles = [d['tile'] for d in request_data]
        game.listValidMovePlayer2All()
        for tile in tiles:
            if tiles in game.listValidMoves:
                if game.place(tile[0], tile[1], tile[2], tile[3]):
                    if game.isvalid:
                        tileonboard.append({'tile': [int(tile[0]), int(tile[1]), int(tile[2]), int(tile[3])]})
                        game.player2.delRack(tile[0], tile[1])
                        game.player2.addTileToRack(game.bag)
                    else:
                        return json.dumps("False")
                else:
                    return json.dumps("False")
            else:
                return json.dumps("False")
        game.player2.point += game.getpoint([[x[2], x[3]] for x in tiles])

        return json.dumps("True")

    return json.dumps("False")




@app.route("/gamereload", methods=['GET'])
@cross_origin()
def gamereload():
    global gridnorme, tileonboard,game
    if request.method == 'GET':
        return tileonboard
if __name__ == "__main__":
    app.run()
