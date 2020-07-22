#!/usr/bin/env python
# coding: utf-8

'''Run post processing on validation data to compare segmentations before and after the post processing
'''

import os
import csv
import numpy as np
import cv2
from scipy import stats
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import efficientnet.tfkeras

# Helper for dirtectory creation
def create_directory(directory):
    '''
    Creates a new folder in the specified directory if the folder doesn't exist.
    INPUT
        directory: Folder to be created, called as "folder/".
    OUTPUT
        New folder in the current directory.
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def post_process(image):
    # Post processing to refine predictions
    image = cv2.medianBlur(image, 5)
    return image


# In[5]:


ROOT_DIR = os.path.abspath("../")
DATASET_NAME = "data0"
DATASET_FOLDER = "npy_data"
#DATASET_FOLDER = "/home/hasib/scratch/npy_data" # use this when on server
DATASET_PATH = os.path.join(ROOT_DIR, "datasets", DATASET_FOLDER)


# In[22]:


data = np.load(DATASET_PATH + '/{}.npz'.format(DATASET_NAME))
train_data = data['name1'] #/255
train_labels = data['name2']

train_data = np.expand_dims(train_data, axis=-1)
train_labels = np.minimum(train_labels, 1)
train_labels = np.expand_dims(train_labels, axis=-1)

# Split into training and validation sets
#x_train = train_data[:2915]
x_test = train_data[2915:]
#y_train = train_labels[:2915]
y_test = train_labels[2915:]

print(x_test.shape, y_test.shape)
print("X Val- max: %s, min: %s" % (np.max(x_test), np.min(x_test)))
print("Y Val- max: %s, min: %s" % (np.max(y_test), np.min(y_test)))


# In[32]:


model_name = 'data0_unet_efb0'
#Load model
weights_path = "../logs/{}/{}.h5".format(model_name, model_name)
model = None
model = load_model(weights_path, compile=False)
preds = model.predict(x=x_test, batch_size=16, verbose=1)
print(preds.shape)


# In[33]:


yp = np.round(preds,0)


# In[34]:


jacard = 0
dice = 0

for i in range(len(y_test)):
    yp_2 = yp[i].ravel()
    y2 = y_test[i].ravel()

    intersection = yp_2 * y2
    union = yp_2 + y2 - intersection

    jacard += (np.sum(intersection)/np.sum(union))  

    dice += (2. * np.sum(intersection) ) / (np.sum(yp_2) + np.sum(y2))


jacard /= len(y_test)
dice /= len(y_test)

print('Jacard Index : '+str(jacard))
print('Dice Coefficient : '+str(dice))


# In[52]:


# After post processing
preds_post =  [post_process(x) for x in preds]
preds_post = np.expand_dims(np.array(preds_post), axis=-1)
print(preds_post.shape)
yp = np.round(preds_post,0)

jacard = 0
dice = 0

for i in range(len(y_test)):
    yp_2 = yp[i].ravel()
    y2 = y_test[i].ravel()

    intersection = yp_2 * y2
    union = yp_2 + y2 - intersection

    jacard += (np.sum(intersection)/np.sum(union))  

    dice += (2. * np.sum(intersection) ) / (np.sum(yp_2) + np.sum(y2))


jacard /= len(y_test)
dice /= len(y_test)

print('Jacard Index after post processing: '+str(jacard))
print('Dice Coefficient after post processing: '+str(dice))


# In[ ]:




