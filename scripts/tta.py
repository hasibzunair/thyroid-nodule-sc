import os, sys
import scipy.misc
from glob import glob
import numpy as np
import random 
import shutil 
import keras
import time
import matplotlib.pyplot as plt
from keras.layers import Dense, Input, Conv2D, Flatten, MaxPool2D, Activation,Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.applications.resnet import ResNet50
from keras.applications.densenet import DenseNet121
from keras.models import load_model
from keras.optimizers import Adam
from keras import optimizers
import pickle
import sys
from keras.preprocessing.image import ImageDataGenerator
import efficientnet.keras as efn
import tensorflow as tf
from tqdm import tqdm
from imgaug import augmenters as iaa   

#-------------- This script need model which can be produced by running train_classifier_efficientnet_augmentation.py--------"

######################  GPU ERROR  #######################

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


################## GPU ERROR ###############################


DATASET_FOLDER = "npy_data"
DATASET_NAME = "data0" # name of the npz file
ROOT_DIR = os.path.abspath("../")

DATASET_PATH = os.path.join(ROOT_DIR, "datasets", DATASET_FOLDER)

data = np.load(DATASET_PATH + '/{}.npz'.format(DATASET_NAME))
train_data = data['name1']
train_labels = data['name3']

# x_train = train_data[:2915]
x_test = train_data[2915:]
# y_train = train_labels[:2915]
y_test = train_labels[2915:]

print(x_test.shape, y_test.shape)
y_test = keras.utils.to_categorical(y_test, 2)
x_test = np.expand_dims(x_test, axis=-1)
print(y_test.shape)

###########TTA###########################

tta_steps = 10
test_datagen = ImageDataGenerator(
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        rotation_range=10.,
        fill_mode='reflect', 
        width_shift_range = 0.1, 
        height_shift_range = 0.1)

test_datagen2 = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-06,
    rotation_range=20,
    width_shift_range=0.0,
    height_shift_range=0.0,
    brightness_range=None,
    shear_range=0.0,
    zoom_range=0.0,
    channel_shift_range=0.0,
    fill_mode="nearest",
    cval=0.0,
    horizontal_flip=True,
    vertical_flip=False,
    rescale=None,
    preprocessing_function=None,
    data_format=None,
    validation_split=0.0,
    dtype=None,
)

model_path = '/logs/data0_final_merged/data0_final_merged.h5'
model = load_model(os.path.join(ROOT_DIR+model_path))
# print(model.summary())

# Make predictions using trained model
# y_pred = model.predict(x_test, verbose=1)
# print("Predictions: ", y_pred.shape)
# x_test /= 255


# x_test = x_test.astype('float32')
# x_test /= 255


predictions_tta = []
for i in tqdm(range(tta_steps)):
    preds = model.predict_generator(test_datagen.flow(x_test, batch_size=16, shuffle=False), steps = len(x_test)/16)
    predictions_tta.append(preds)

final_pred = np.mean(predictions_tta, axis = 0)

# print(y_pred.shape, final_pred.shape)

# Convert ground truth to column values
y_test_flat = np.argmax(y_test, axis=1)
print("After flattening ground truth: ", y_test_flat.shape)


# Get labels from predictions
y_pred_flat = np.array([np.argmax(pred) for pred in final_pred]) 
print("Binarize probability values: ", y_pred_flat.shape)

assert y_pred_flat.shape == y_test_flat.shape, "Shape mismatch!"


# In[12]:


# Sanity check

print(y_test.shape, y_test_flat.shape, final_pred.shape, y_pred_flat.shape)


# In[15]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

# Accuracy
acc = accuracy_score(y_test_flat, y_pred_flat) * 100
print("Accuracy :", acc)


# In[16]:


# Classification report

confusion_mtx = confusion_matrix(y_test_flat, y_pred_flat) 
print(confusion_mtx)
target_names = ['0', '1']
print(classification_report(y_test_flat, y_pred_flat, target_names=target_names))


# In[17]:


tn, fp, fn, tp = confusion_matrix(y_test_flat, y_pred_flat).ravel()
print("tn, fp, fn, tp: ", tn, fp, fn, tp)


# In[18]:


sensitivity = tp /(tp + fn)
print("Sensitivity: ",sensitivity * 100)


# In[31]:


from sklearn.metrics import roc_auc_score
print('Area under ROC curve : ', roc_auc_score(y_test, final_pred) *100 )


# In[21]:


from sklearn.metrics import f1_score
F1_score = f1_score(y_test_flat, y_pred_flat, average='weighted')
print("F1 score: ", F1_score)








