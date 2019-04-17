import numpy as np
import time
from getkeys import key_check
from FIFA import FIFA
from exp_repl_actor_critic import ExperienceReplay
import random 
from keras.layers.core import Dense
from keras.models import Sequential,Model
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Input,GlobalAveragePooling2D,Flatten
from keras.layers.merge import Add, Multiply
import keras.backend as K

import tensorflow as tf

class Train(object):

    def __init__(self,sess):
        self.sess = sess        
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = .995
        self.gamma = .95
        self.tau   = .125
        self.exp_replay = ExperienceReplay()
        #Actor Model

        self.actor_state_input, self.actor_model = self.create_actor_model()
        _, self.target_actor_model = self.create_actor_model()

        self.actor_critic_grad = tf.placeholder(tf.float32, 
            [None, 4]) 
        
        actor_model_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output, 
            actor_model_weights, -self.actor_critic_grad) # dC/dA (from actor)
        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

        #Critic model     

        self.critic_state_input, self.critic_action_input, \
            self.critic_model = self.create_critic_model()
        _, _, self.target_critic_model = self.create_critic_model()

        self.critic_grads = tf.gradients(self.critic_model.output, 
            self.critic_action_input) # where we calcaulte de/dC for feeding above
        
        # Initialize for later gradient calculations
        self.sess.run(tf.initialize_all_variables())

    

    def create_actor_model(self):
        state_input = Input(shape=(7,7,512))
        h1 = GlobalAveragePooling2D()(state_input)
        h1 = Dense(256, activation='relu')(h1)
        h2 = Dense(512, activation='relu')(h1)
        h3 = Dense(256, activation='relu')(h2)
        output = Dense(4, activation='relu')(h3)
        
        model = Model(input=state_input, output=output)
        adam  = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, model

    def create_critic_model(self):
        state_input = Input(shape=(7,7,512))
        state_h1 = GlobalAveragePooling2D()(state_input)
        state_h1 = Dense(256, activation='relu')(state_h1)
        state_h2 = Dense(512)(state_h1)
        
        action_input = Input(shape=(4,))
        action_h1    = Dense(512)(action_input)
        
        merged    = Add()([state_h2, action_h1])
        merged_h1 = Dense(256, activation='relu')(merged)
        output = Dense(1, activation='relu')(merged_h1)
        model  = Model(input=[state_input,action_input], output=output)
        
        adam  = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, action_input, model

    def train_actor_critic(self,inputs):
        self._train_actor(inputs)
        self._train_critic(inputs)

    def _train_actor(self, samples):
        
            for sample in samples:
                cur_state, action, reward, new_state, _ = sample
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    predicted_action = self.actor_model.predict(cur_state)
                    grads = sess.run(self.critic_grads, feed_dict={
                        self.critic_state_input:  cur_state,
                        self.critic_action_input: predicted_action
                    })[0]

                    sess.run(self.optimize, feed_dict={
                        self.actor_state_input: cur_state,
                        self.actor_critic_grad: grads
                    })
            
    def _train_critic(self, samples):

            for sample in samples:
                
                cur_state, action, reward, new_state, done = sample
                if not done:
                    print("not done")
                    target_action = self.target_actor_model.predict(new_state)
                    future_reward = self.target_critic_model.predict(
                        [new_state, target_action])[0][0]
                    reward += self.gamma * future_reward
                reward = np.asarray([reward])

                print(reward)
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    self.critic_model.fit([cur_state, action], reward, verbose=0)
     

    def control_bot(self,game,epochs):
    # Train
        with self.sess.as_default() as sess:
            cou = 0
      
            paused = True
            gamma = 0.9 
            self.num_actions = 4  # [ shoot_low, shoot_high, left_arrow, right_arrow]
            max_memory = 1000  # Maximum number of experiences we are storing
            self.batch_size = 4  # Number of experiences we use for training per batch
           
            win_cnt = 0
            loss_cnt = 0
            loss = 0
            # We want to keep track of the progress of the AI over time, so we save its win count history
            win_hist = []
            game_over = False

            # Epochs is the number of games we play
            for e in range(epochs):
                # epsilon = 4 / ((e + 1) ** (1 / 2))
                epsilon = 0.1
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
                        input_tm1 = input_t #shape output of VGG feature extractor (1,7,7,512)
                        #take a random action
                        if random.choice([1,2,3,4,5,6,7,8,9,10]) in [1,5,10]:
                            print("inside random")
                            cou+=1
                            action = int(np.random.randint(0, 4, size=1))

                        else:
                            print("inside predict")
                            with tf.Session() as sess:
                                sess.run(tf.global_variables_initializer())
                                q = self.actor_model.predict(input_tm1)
                            action = np.argmax(q[0])
                            print(q[0])
                            input_t, reward, game_over = game.act(action)
                            print("reward calculated - " + str(reward)) 
                            self.exp_replay.remember([input_tm1, q, reward, input_t ,game_over])

                        # Load batch of experiences
                        inputs = self.exp_replay.get_batch(batch_size=4)
                      
                        self.train_actor_critic(inputs)
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

