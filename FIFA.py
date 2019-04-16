import numpy as np
import pytesseract as pt
import cv2
from PIL import Image
from grabscreen import grab_screen
from directkeys import *
import ConvolutionalNN as cnn
import random

count = 0

class FIFA(object):
    reward = 0

    def __init__(self):
        self.reset()
        self.hist = []

    def game_over(self, action):
        is_over = True if action in [0, 1] else False
        return is_over    

    def observe(self):
        print('\n\nobserve')
        # get current screen using screen-grab
        screen = grab_screen(region=None)
        screen = screen

        # if freekick drill is over, then restart drill
        restart_button = screen
        i = Image.fromarray(restart_button.astype('uint8'), 'RGB')
        restart_text = pt.image_to_string(i)
               

        if "RETRV DRILL" in restart_text:
            self.reward = self.get_final_drill_score()
            if self.reward != 0:
                file = open("model_epoch1000/history.txt","a")
                file.write(str(self.reward))
                file.write("\n")
                file.close()
            # press enter key
            print('pressing enter, reset reward')
            self.reward = 0
            PressKey(leftarrow)
            time.sleep(0.4)
            ReleaseKey(leftarrow)
            PressKey(leftarrow)
            time.sleep(0.4)
            ReleaseKey(leftarrow)
            PressKey(enter)
            time.sleep(0.4)
            ReleaseKey(enter)
            time.sleep(2)
            screen = grab_screen(region=None)
            screen = screen
            self.hist.append(self.reward)
            if len(self.hist) > 50:
                file = open("model_epoch1000/history.txt","w")
                file.write(str(self.hist))
                self.hist = []
        state = cnn.get_image_content(screen)
        return state

    def get_final_drill_score(self):
        screen = grab_screen(region=None)
        crop_img = self.crop_image(screen, 550, 235, 50, 50)
        i = Image.fromarray(crop_img.astype('uint8'), 'RGB')
        ocr_result = pt.image_to_string(i)
        print(str(ocr_result))
        try:
            ingame_reward = int(''.join(c for c in ocr_result if c.isdigit()))
            if self.reward < ingame_reward:
                self.reward = ingame_reward
            print("drill final score - " + str(self.reward))
        except Exception as e:
            print(e)
            pass
        return self.reward                

    # calculating rewards by grabbing screen and cropping only drill points        
    def calculate_rewards(self,action):
        screen = grab_screen(region=None)
        crop_img = self.crop_image(screen, 1125, 40, 65, 55)
        return self.get_reward_by_ocr(crop_img,action) 
        
    # cropping image by start x and start y with width and height    
    def crop_image(self, image, start_x,start_y, crop_x, crop_y):
        crop_img = image[start_y:start_y + crop_y, start_x:start_x+crop_x]
        return crop_img

    def get_reward_by_ocr(self, reward_screen, action):

        # In fifa 18, we give rewards based on player's freekick score after every shoot. 
        # Ideally, it give 1000 or more than 1000 when it hits one of the four targets.
        # It gives approximate 800 - 999 if it hits near target.
        # it gives 500- 700 if player just scores no where near target.
        # it gives 200 if the shot hits the goal post.
        ingame_reward = 0
        i = Image.fromarray(reward_screen.astype('uint8'), 'RGB')
        ocr_result = pt.image_to_string(i)
        try:
            for c in ocr_result:
                if not c.isdigit():
                   raise Exception("invalid score")
            ingame_reward = int(''.join(c for c in ocr_result if c.isdigit()))
            temp_reward = ingame_reward
            print("ingame_reward - " + str(ingame_reward) + " reward - " + str(self.reward) + " action - " + str(action))
            if (ingame_reward - self.reward) >= 1000 and self.game_over(action):
                ingame_reward = 5
            elif (ingame_reward - self.reward) < 1000 and (ingame_reward - self.reward) > 500 and self.game_over(action):
                ingame_reward = 1    
            elif (ingame_reward - self.reward) < 500 and (ingame_reward - self.reward) > 0 and self.game_over(action):
                ingame_reward = 1
            elif (ingame_reward - self.reward) == 0 and self.game_over(action):
                ingame_reward = -10
            else:
                ingame_reward = 0    
            self.reward = temp_reward
        except Exception as e:
            print(e)
            reward_list = [0.5, -0.5]
            value = random.choice(reward_list)
            print(str(value))
            ingame_reward = value
            pass
        return ingame_reward
            

    def act(self, action):
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

        # wait until some time after taking action for more power and variations in shots
        if action in [0, 1]:
            time.sleep(5)
        else:
            time.sleep(1)

        ingame_reward = self.calculate_rewards(action)
        is_game_over = self.game_over(action)      
        return self.observe(), ingame_reward, is_game_over

    def reset(self):
        return
