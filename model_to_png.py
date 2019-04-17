from keras.utils import plot_model
from keras.models import model_from_json

from keras.layers.core import Dense
from keras.models import Sequential,Model
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Input,GlobalAveragePooling2D,Flatten
from keras.layers.merge import Add, Multiply



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
    plot_model(loaded_model, to_file='model.png',show_shapes=True)
    return loaded_model

def create_actor_model():
        state_input = Input(shape=(7, 7, 512))
        h1 = GlobalAveragePooling2D()(state_input)
        h1 = Dense(256, activation='relu')(h1)
        h2 = Dense(512, activation='relu')(h1)
        h3 = Dense(256, activation='relu')(h2)
        output = Dense(4, activation='relu')(h3)

        model = Model(input=state_input, output=output)
        adam = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        plot_model(model, to_file='actor_model.png',show_shapes=True)



def create_critic_model():
        state_input = Input(shape=(7, 7, 512))
        state_h1 = GlobalAveragePooling2D()(state_input)
        state_h1 = Dense(256, activation='relu')(state_h1)
        state_h2 = Dense(512)(state_h1)

        action_input = Input(shape=(4,))
        action_h1 = Dense(512)(action_input)

        merged = Add()([state_h2, action_h1])
        merged_h1 = Dense(256, activation='relu')(merged)
        output = Dense(1, activation='relu')(merged_h1)
        model = Model(input=[state_input, action_input], output=output)

        adam = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        plot_model(model, to_file='critic_model.png',show_shapes=True)


load_model()

create_actor_model()
create_critic_model()
