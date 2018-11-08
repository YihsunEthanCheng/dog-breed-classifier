# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 01:07:03 2018

@author: Ethan Cheng
"""
from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob

# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset('dogImages/train')
valid_files, valid_targets = load_dataset('dogImages/valid')
test_files, test_targets = load_dataset('dogImages/test')

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]

# print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))

#%%

import random
random.seed(8675309)

# load filenames in shuffled human dataset
human_files = np.array(glob('lfw\**\*.jpg', recursive = True))
random.shuffle(human_files)

# print statistics about the dataset
print('There are %d total human images.' % len(human_files))

#%%
import cv2                
import matplotlib.pyplot as plt                        

# extract pre-trained face detector
face_haarCascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
face_lbpCascade = cv2.CascadeClassifier('haarcascades/lbpcascade_frontalface.xml')

# load color (BGR) image
img = cv2.imread(human_files[np.random.randint(len(human_files))])
# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find faces in image
faces = face_haarCascade.detectMultiScale(gray)

# print number of faces detected in the image
print('Number of faces detected:', len(faces))

# get bounding box for each detected face
for (x,y,w,h) in faces:
    # add bounding box to color image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.clf()
plt.imshow(cv_rgb)
plt.show()

#%%
# returns "True" if face is detected in image stored at img_path
def face_detector(img_path, detector):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray)
    return len(faces) > 0

#%%
    
human_files_short = human_files[:100]
dog_files_short = train_files[:100]
# Do NOT modify the code above this line.

## TODO: Test the performance of the face_detector algorithm 
## on the images in human_files_short and dog_files_short.
#%%
h_res = []
for h_file in human_files_short:
    for detector in [face_haarCascade, face_lbpCascade]:
        h_res += [[face_detector(h_file, face_haarCascade), face_detector(h_file, face_lbpCascade)]]
        #%%
human_rec_rate = np.sum(h_res, axis = 0)/len(h_res)    

#%%
d_res = []
for d_file in dog_files_short:
    for detector in [face_haarCascade, face_lbpCascade]:
        d_res += [[face_detector(d_file, face_haarCascade), face_detector(d_file, face_lbpCascade)]]
dogFAR = np.sum(d_res, axis=0)/len(d_res)
print(dogFAR) 