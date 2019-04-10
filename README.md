# Freekick_Bot_FIFA18


This project is a simple setup which simulates FIFA 18 freekicks.

It tries to train itself by performing various actions (shooting high, shooting low, moving the player left and moving the player on the right)

It also simulates restarting the drill after the drill is over for continuous interrupted training.

Progress on project

We have implemented a deep Q learning network which learns the conditions and state from the screen shots we grab screen and pass it to VGG16 a famous deep learning network which comes with keras.applications to extract features out of the screenshot that we capture. We pass this features to a dense neural network with 3 layers to generate Q value for all four of our actions which are shoot_high, shoot_low, left arrow and right arrow. We select the action with highest Q value and simulate that key press. We then grab screen and run OCR to extract the reward of that shot taken by DQN. We are currently using FIFA generated points.

In fifa 18, we give rewards based on player's freekick score after every shoot. Ideally, it give 1000 or more than 1000 when it hits one of the four targets.It gives approximate 800 - 999 if it hits near target.it gives 500- 700 if player just scores no where near target.It gives 200 if the shot hits the goal post.

We trained it for various amounts of time and were getting somewhat okay results with time. We also made sure that this results are just not because of random selection and decayed the probability with which the controller selects the actions. As time passes on it starts using model generated actions and reduces dependence on random actions. We also used feature extraction using Deep neural network as opposed to passing the whole image to the Q_function neural network. 

We are focusing on our experience replay technique right now. We are currently storing a large amount of previous screens in order to store and explain the model temporal relation between frames and Q score. We want to experiment on LSTMs as opposed to this memory consuming Experience Replay logic that is generally used in Deep Q Learning.





