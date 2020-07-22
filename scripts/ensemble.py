# %%
"""
# Trains a baseline U-Net
"""


# %%
# Import libraries
import os
import time
import matplotlib.pyplot as plt

#matplotlib inline
import numpy as np
import time
from tqdm import tqdm
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau
import keras
from keras.layers import Input
from keras.models import Model
from keras.utils import generic_utils
from keras import callbacks
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import efficientnet.tfkeras
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras import layers
from scipy import ndimage, misc
# Go back one step to read module
import sys
sys.path.insert(0,"..") 

import segmentation_models as sm
import data_utils
import classes
import models as M
import losses as l
import metrics

model_name1 ='data0_Cascaded_efb5_loss_augmentation_8_adversarial_nadam_different_AUG' #submission2

model_name2 = 'data0_Cascaded_efb3_loss_augmentation_8_adversarial' #submission1
#model_name2 = 'data0_Cascaded_efb3_loss_augmentation_8_adversarial_nadam_different_AUG' #submission0
# %%
# Name data and config types
DATASET_NAME = "data0" # name of the npz file
ROOT_DIR = os.path.abspath("../")
DATASET_FOLDER = "npy_data"
#DATASET_FOLDER = "/home/hasib/scratch/npy_data" # use this when on server
DATASET_PATH = os.path.join(ROOT_DIR, "datasets", DATASET_FOLDER)



data = np.load(DATASET_PATH + '/{}.npz'.format(DATASET_NAME))
train_data = data['name1']/255
train_labels = data['name2']

train_data = np.expand_dims(train_data, axis=-1)
train_labels = np.minimum(train_labels, 1)
train_labels = np.expand_dims(train_labels, axis=-1)

# %%
"""
## Split data and train
"""

# %%
# Split into training and validation sets
x_train = train_data[:2915]
x_test = train_data[2915:]
y_train = train_labels[:2915]
y_test = train_labels[2915:]


print("Train and validate on -------> ", x_train.shape, x_test.shape, y_train.shape, y_test.shape)

print("\n\nX Train- max: %s, min: %s" %(np.max(x_train), np.min(x_train)))
print("Y Train- max: %s, min: %s" % (np.max(y_train), np.min(y_train)))
print("X Val- max: %s, min: %s" % (np.max(x_test), np.min(x_test)))
print("Y Val- max: %s, min: %s" % (np.max(y_test), np.min(y_test)))


#Load model1
weights_path = "../logs/{}/{}.h5".format(model_name1, model_name1)
model1 = None
model1 = load_model(weights_path, compile=False)
y_pred1 = model1.predict(x=x_test, batch_size=16, verbose=1)

weights_path = "../logs/{}/{}.h5".format(model_name2, model_name2)
model2 = None
model2 = load_model(weights_path, compile=False)
y_pred2 = model2.predict(x=x_test, batch_size=16, verbose=1)



#Check Jaccard

operation_point, _, _, accuracy, specificity, sensitivity, dice, jaccard = data_utils.use_operating_points(0.5, y_test.flatten(), y_pred1.flatten())

print(
    "\nModel1: OP:{:.4f}, Accuracy :{:.4f}, Sensitivity:{:.4f}, Specificity: {:.4f}, Dice: {:.4f}, Jaccard: {:.4f}".format(
        operation_point, accuracy, sensitivity, specificity, dice, jaccard))


operation_point, _, _, accuracy, specificity, sensitivity, dice, jaccard = data_utils.use_operating_points(0.5, y_test.flatten(), y_pred2.flatten())

print(
    "\nModel2: OP:{:.4f}, Accuracy :{:.4f}, Sensitivity:{:.4f}, Specificity: {:.4f}, Dice: {:.4f}, Jaccard: {:.4f}".format(
        operation_point, accuracy, sensitivity, specificity, dice, jaccard))


y_pred3 = np.logical_and(y_pred1>=0.5, y_pred2>=0.5)

operation_point, _, _, accuracy, specificity, sensitivity, dice, jaccard = data_utils.use_operating_points(0.5, y_test.flatten(), y_pred3.flatten())

print(
    "\nAnd: OP:{:.4f}, Accuracy :{:.4f}, Sensitivity:{:.4f}, Specificity: {:.4f}, Dice: {:.4f}, Jaccard: {:.4f}".format(
        operation_point, accuracy, sensitivity, specificity, dice, jaccard))


y_pred4 = np.logical_or(y_pred1>=0.5, y_pred2>=0.5)

operation_point, _, _, accuracy, specificity, sensitivity, dice, jaccard = data_utils.use_operating_points(0.5, y_test.flatten(), y_pred4.flatten())

print(
    "\nOr: OP:{:.4f}, Accuracy :{:.4f}, Sensitivity:{:.4f}, Specificity: {:.4f}, Dice: {:.4f}, Jaccard: {:.4f}".format(
        operation_point, accuracy, sensitivity, specificity, dice, jaccard))


#post processing
y_pred5 =  ndimage.median_filter(y_pred4, size=10)

operation_point, _, _, accuracy, specificity, sensitivity, dice, jaccard = data_utils.use_operating_points(0.5, y_test.flatten(), y_pred5.flatten())

print(
    "\nOr: OP:{:.4f}, Accuracy :{:.4f}, Sensitivity:{:.4f}, Specificity: {:.4f}, Dice: {:.4f}, Jaccard: {:.4f}".format(
        operation_point, accuracy, sensitivity, specificity, dice, jaccard))



print('debug')