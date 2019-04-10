import numpy as np
import time
from getkeys import key_check
from FIFA import FIFA
from experience_replay import ExperienceReplay

paused = True

num_actions = 4  # [ shoot_low, shoot_high, left_arrow, right_arrow]
max_memory = 1000  # Maximum number of experiences we are storing
batch_size = 4  # Number of experiences we use for training per batch
exp_replay = ExperienceReplay(max_memory=max_memory)

def save_model(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("model_epoch1000/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_epoch1000/model.h5")
    # print("Saved model to disk")

def control_bot(game,epochs, model):
    # Train
    win_cnt = 0
    loss_cnt = 0
    matka = 0
    loss = 0
    # We want to keep track of the progress of the AI over time, so we save its win count history
    win_hist = []
    game_over = False
    # Epochs is the number of games we play
    for e in range(epochs):
        # epsilon = 4 / ((e + 1) ** (1 / 2))
        epsilon = 0.9
        # Resetting the game
        game.reset()
        game_over = False
        # get tensorflow running first to acquire cudnn handle
        input_t = game.observe()
        if e == 0:
            paused = True
            print('Training is paused. Press p once game is loaded and is ready to be played.')
        else:
            paused = False
        while not game_over:
            if not paused:
                # The learner is acting on the last observed game screen
                # input_t is a vector containing representing the game screen
                input_tm1 = input_t

                if np.random.rand() <= epsilon:
                    # Eat something random from the menu
                    action = int(np.random.randint(0, num_actions, size=1))
                # else:
                #     print(input_tm1.shape)
                #     q = model.predict(input_tm1)
                #     print(q.shape)
                #     action = np.argmax(q[0])
                #     print(q[0])
                # print(action)
                input_t, reward, game_over = game.act(action)
                print("reward calculated - " + str(reward)) 

                # print("inside FIFA",input_tm1.shape,action,reward,input_t.shape)
                # exp_replay.remember([input_tm1, action, reward, input_t], game_over)

                # # Load batch of experiences
                # inputs, targets = exp_replay.get_batch(model, batch_size=1)

                # # train model on experiences
                # batch_loss = model.train_on_batch(inputs, targets)

                # # print(loss)
                # loss += batch_loss
            keys = key_check()
            if 'P' in keys:
                if paused:
                    paused = False
                    print('unpaused!')
                    time.sleep(1)
                else:
                    print('Pausing!')
                    paused = True
                    time.sleep(1)
            elif 'O' in keys:
                print('Quitting!')
                return

        save_model(model)
        win_hist.append(win_cnt)