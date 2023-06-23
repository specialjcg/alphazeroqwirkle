import pickle


game_action_prob = pickle.load(open('gameActionProb.pkl', 'rb'))
# Set it as a global constant
GLOBAL_GAME_ACTION_PROB = game_action_prob
