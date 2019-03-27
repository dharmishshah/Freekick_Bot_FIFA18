import numpy as np
import time
from getkeys import key_check
from FIFA import FIFA

paused = True

def control_bot(game,epochs=1000):
    # Train
    num_actions = 4
    game_over = False
    # Epochs is the number of games we play
    for e in range(epochs):
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
                # Select a random action
                action = int(np.random.randint(0, num_actions, size=1))
                print('random action')
                game.act(action)
                
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


game = FIFA()
control_bot(game)