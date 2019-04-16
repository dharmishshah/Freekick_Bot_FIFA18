import numpy as np
import os
import math

class ExperienceReplay(object):

    def __init__(self, max_memory=1000, discount=.3):
        
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, states):
        # Save a state to memory
        self.memory.append(states)
        # We don't want to store infinite memories, so if we have too many, we just delete the oldest one
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, batch_size=10):

        len_memory = len(self.memory)
        # Calculate the number of actions that can possibly be taken in the game( we are considering 4 actions)

        num_actions = 4
        inputs = np.zeros((min(len_memory, batch_size), 7,7,512))
       
        targets = np.zeros((inputs.shape[0], num_actions))

        
        samples = []
        # We draw states to learn from randomly
        for i, idx in enumerate(np.random.randint(0, len_memory,
                                                  size=inputs.shape[0])):
            samples.append(self.memory[idx])

            # We also need to know whether the game ended at this state
            
        return samples
