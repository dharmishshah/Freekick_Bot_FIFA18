import os
import time
import cv2
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np



model = VGG16(weights='imagenet', include_top=False)	


def get_image_content(image):
	image = cv2.resize(image,dsize = (224,224),interpolation = cv2.INTER_CUBIC)
	image = np.expand_dims(image,axis=0)
	img_data = preprocess_input(image)
	vgg16_feature = model.predict(img_data)
	return vgg16_feature

	 
	