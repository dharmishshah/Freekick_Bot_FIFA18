import numpy as np
import pytesseract as pt
import cv2
from PIL import Image
from grabscreen import grab_screen
from directkeys import *



#Game class that performs the action grabs screen and restarts the drill.

class FIFA(object):
    reward = 0

    def __init__(self):
        self.reset()


    #observes the game state if the drill is finished the retry button is pressed.
    def observe(self):
        print('\n\nobserve')
        # get current state s from screen using screen-grab
        screen = grab_screen(region=None)
        screen = screen

        # if drill over, restart drill and take screenshot again
        restart_button = screen
        i = Image.fromarray(restart_button.astype('uint8'), 'RGB')
        restart_text = pt.image_to_string(i)
        if "RETRV DRILL" in restart_text:
            # press enter key
            print('pressing enter, reset reward')
            self.reward = 0
            PressKey(leftarrow)
            time.sleep(0.4)
            ReleaseKey(leftarrow)
            PressKey(enter)
            time.sleep(0.4)
            ReleaseKey(enter)
            time.sleep(2)
            screen = grab_screen(region=None)
            screen = screen

    def act(self, action):
        # [ shoot_low, shoot_high, left_arrow, right_arrow ]
        display_action = ['shoot_low', 'shoot_high', 'left_arrow', 'right_arrow']
        print('action: ' + str(display_action[action]))

        keys_to_press = [[D], [D], [leftarrow], [rightarrow]]
        # need to keep all keys pressed for some time before releasing them otherwise fifa considers them as accidental
        # key presses.
        for key in keys_to_press[action]:
            PressKey(key)
        time.sleep(0.05) if action == 0 else time.sleep(0.2)
        for key in keys_to_press[action]:
            ReleaseKey(key)

        # wait until some time after taking action
        if action in [0, 1]:
            time.sleep(5)
        else:
            time.sleep(1)

        return self.observe()

    def reset(self):
        return
