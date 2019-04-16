import numpy as np
import os
import math

class ExperienceReplay(object):

    def __init__(self, max_memory=1000, discount=.3):
        
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, states, game_over):
        # Save a state to memory
        self.memory.append([states, game_over])
        # We don't want to store infinite memories, so if we have too many, we just delete the oldest one
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=10):

        len_memory = len(self.memory)
        # Calculate the number of actions that can possibly be taken in the game( we are considering 4 actions)

        num_actions = 4
        env_dim = self.memory[0][0][0].shape[1]
        inputs = np.zeros((min(len_memory, batch_size), 7,7,512))
       
        targets = np.zeros((inputs.shape[0], num_actions))

        

        # We draw states to learn from randomly
        for i, idx in enumerate(np.random.randint(0, len_memory,
                                                  size=inputs.shape[0])):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            # We also need to know whether the game ended at this state
            game_over = self.memory[idx][1]
            
            # add the state s to the input
            inputs[i:i + 1] = state_t

            # First we fill the target values with the predictions of the model.
            # They will not be affected by training (since the training loss for them is 0)
            targets[i] = model.predict(state_t)[0]

            """
            If the game ended, the expected reward Q(s,a) should be the final reward r.
            Otherwise the target value is r + gamma * max Q(s’,a’)
            """
            #  Here Q_sa is max_a'Q(s', a')
            Q_sa = np.max(model.predict(state_tp1)[0])

            # if the game ended, the reward is the final reward
            if game_over and reward_t != 0:
                targets[i,action_t] = reward_t
            elif not game_over:
                targets[i, action_t] = reward_t + self.discount * Q_sa
            else:
                targets[i] = targets[i]
            # if game_over or reward_t > 0:  # if game_over is True
            #     print("game_over")
            #     targets[i, action_t] = targets[i,action_t]
            # elif not game_over:
            #     print("game not over")
            #     # r + gamma * max Q(s’,a’)
            #     targets[i, action_t] = reward_t + self.discount * Q_sa
            # else:
            #     print("inside targets")
            #     targets[i] = targets[i]
        
        return inputs, targets
