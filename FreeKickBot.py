import numpy as np
import pytesseract as pt
from keras.layers.core import Dense
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Flatten
from keras.optimizers import sgd
from matplotlib import pyplot as plt
from FIFA import FIFA
from train import control_bot
from keras.layers import Flatten

def baseline_model(grid_size, num_actions, hidden_size):
    # setting up the model with keras
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(512,activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(4))
    model.compile(sgd(lr=.01),loss="mse")
    return model


def load_model():
    # load json and create model
    json_file = open('model_epoch1000/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model_epoch1000/model.h5")
    print("Loaded model from disk")
    loaded_model.compile(loss='mse', optimizer='sgd')
    return loaded_model


# model = baseline_model(grid_size=128, num_actions=4, hidden_size=512)
model = load_model()
# model.summary()

game = FIFA()
# Number of games played in training, I found the model needs about 4,000 games till it plays well
epoch = 1000  
train_mode = 1

if train_mode == 1:
    # Training the model
    hist = control_bot(game,epoch, model)
    print("Training done")