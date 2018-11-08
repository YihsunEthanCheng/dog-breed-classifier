# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 04:28:13 2018

@author: Ethan Cheng
"""
from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
#from keras.preprocessing import image                  
#from tqdm import tqdm
import matplotlib.pyplot as plt

#### get data
# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

#% load train, test, and validation datasets
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

import numpy as np
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dropout, Dense
from keras.models import Sequential

#%%
netID = 'InceptionV3'
netID = 'Resnet50'
bottleneck_features = np.load('bottleneck_features/Dog'+netID+'Data.npz')
train_Resnet50 = bottleneck_features['train']
valid_Resnet50 = bottleneck_features['valid']
test_Resnet50 = bottleneck_features['test']

#%%
Resnet50_model = Sequential()
Resnet50_model.add(GlobalAveragePooling2D(input_shape=train_Resnet50.shape[1:]))
Resnet50_model.add(Dropout(0.25))
Resnet50_model.add(Dense(133, activation='softmax'))

Resnet50_model.summary()

Resnet50_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

#%%
from keras.callbacks import ModelCheckpoint  

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.Resnet50.hdf5', 
                               verbose=1, save_best_only=True)

Resnet50_model.fit(train_Resnet50, train_targets, 
          validation_data=(valid_Resnet50, valid_targets),
          epochs=40, batch_size=20, callbacks=[checkpointer], verbose=1)

#%%

Resnet50_model.load_weights('saved_models/weights.best.Resnet50.hdf5')

# get index of predicted dog breed for each image in test set
Resnet50_predictions = [np.argmax(Resnet50_model.predict(np.expand_dims(feature, axis=0))) for feature in test_Resnet50]

# report test accuracy
test_accuracy = 100*np.sum(np.array(Resnet50_predictions)==np.argmax(test_targets, axis=1))/len(Resnet50_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)

#%%
from extract_bottleneck_features import *
from keras.preprocessing import image                  
def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

#%%
def predict_dog_breed(img_path):
    img_tensor = path_to_tensor(img_path)
    f_tensor = extract_Resnet50(img_tensor)
    fvec = np.array([ np.mean(f_tensor[:,:,:,i]) for i in range(f_tensor.shape[-1])])
    nameid = np.argmax(Resnet50_model.predict(fvec[None, None,None,:]))
    return dog_names[nameid]


#%%
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return faces, img

#%%
from keras.applications.resnet50 import ResNet50

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')
    
from keras.applications.resnet50 import preprocess_input, decode_predictions

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 

#%%    

def Recognize_human_dog_breed(img_path):
    greetings = 'Hello ' 
    verdict = 'You look like a '
    faces, img = face_detector(img_path)
    for (x,y,w,h) in faces:
        # add bounding box to color image
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    if len(faces) > 0:
        greetings += 'human !!'
    else:
        greetings += 'dog !!'
        
    verdict += predict_dog_breed(img_path)
    print(greetings)
    plt.imshow(img)
    print(verdict)
    return img, greetings, verdict

#%%
    
my_file = ['C:\_cloud\dl_nano\dog-project\images\kim.jpg', 
           'C:\_cloud\dl_nano\dog-project\images\obama.jpg',
           'C:\_cloud\dl_nano\dog-project\images\kardashian.jpg',
           'C:\_cloud\dl_nano\dog-project\images\trump_0.jpg',
           'C:\_cloud\dl_nano\dog-project\images\trump_1.jpg']
