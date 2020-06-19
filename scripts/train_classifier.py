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

# Go back one step to read module
import sys
sys.path.insert(0,"..") 

import data_utils
import classes

CFG_NAME = "vgg_classifier" # name of the architecture/configuration


DATASET_FOLDER = "npy_data"
DATASET_NAME = "data0" # name of the npz file
ROOT_DIR = os.path.abspath("../")
DATASET_PATH = os.path.join(ROOT_DIR, "datasets", DATASET_FOLDER)
EXPERIMENT_NAME = "{}_{}".format(DATASET_NAME, CFG_NAME)

# In[2]:

if not os.path.exists(os.path.join(ROOT_DIR, "logs")):
    os.mkdir(os.path.join(ROOT_DIR, "logs"))

# Make log path to store all results
LOG_PATH = os.path.join(ROOT_DIR, "logs", EXPERIMENT_NAME)

if not os.path.exists(LOG_PATH):
    os.mkdir(LOG_PATH)

    
# Open log file
log_file = open("{}/{}_log.out".format(LOG_PATH, EXPERIMENT_NAME), 'w')
sys.stdout = log_file

# In[3]:


# Load data
data = np.load(DATASET_PATH + '/{}.npz'.format(DATASET_NAME))
train_data = data['name1']
train_labels = data['name3']

train_data = np.expand_dims(train_data, axis=-1)
print(train_data.shape, train_labels.shape)


# Split into training and validation sets
x_train = train_data[:2915]
x_test = train_data[2915:]
y_train = train_labels[:2915]
y_test = train_labels[2915:]

y_train = keras.utils.to_categorical(y_train, 2)
y_test = keras.utils.to_categorical(y_test, 2)

print("Train and validate on -------> ", x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# In[4]:


from keras.layers import Input, Conv2D
from keras.models import Model

def classification_network():
    
    #base_model = DenseNet121(weights='imagenet',include_top=False,pooling='avg',input_shape=(256, 256, 3))
    base_model = VGG16(weights='imagenet',include_top=False,pooling='avg',input_shape=(256, 256, 3))
    #base_model = ResNet50(weights='imagenet',include_top=False,pooling='avg',input_shape=(256, 256, 3))
    
    inp = Input(shape=(256, 256, 1))
    l1 = Conv2D(3, (1, 1))(inp) # map N channels data to 3 channels
    out = base_model(l1)
    
    # Add FC layer
    predictions = Dense(2, activation='softmax', trainable=True)(out) 
    
    for layer in base_model.layers:
        layer.trainable=True
        
    model = Model(inputs=[inp], outputs=[predictions])
        
    # Optimzer and loss
    optim = optimizers.Adadelta(lr=0.001)
    loss_func = 'binary_crossentropy' 
    
    model.compile(optimizer=optim, loss=loss_func, metrics=['accuracy'])
    return model

model = None
model = classification_network()
model.summary()


# In[5]:


# Define callbacks for learning rate scheduling, logging and best checkpoints saving
callbacks = [
    keras.callbacks.ModelCheckpoint('{}/{}.h5'.format(LOG_PATH, EXPERIMENT_NAME), monitor='val_loss', save_best_only=True, mode='min'),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, verbose=1, patience=5, mode='min'), ## new_lr = lr * factor 
    keras.callbacks.CSVLogger('{}/training.csv'.format(LOG_PATH)),
    keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, verbose=1, patience=15, mode='min', restore_best_weights=True)]


# In[7]:


start_time = time.time()


model.fit(x=x_train, y=y_train, 
        batch_size=32, 
        epochs=200, 
        verbose=2, 
        validation_data=(x_test,y_test),
        shuffle=True, callbacks=callbacks)


end_time = time.time()
print("--- Time taken to train : %s hours ---" % ((end_time - start_time)//3600))


# In[8]:


# Plot and save accuravy loss graphs individually
def plot_loss(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))
    plt.plot(epochs, loss, 'g')
    plt.plot(epochs, val_loss, 'y')
    #plt.title('Training and validation loss')
    plt.ylabel('Loss %')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.grid(True)
    plt.savefig('{}/{}_loss.jpg'.format(LOG_PATH, EXPERIMENT_NAME), dpi=100)
    plt.close('all')
    #plt.show()
    
def plot_acc(history):
    acc = history.history['accuracy']
    vacc = history.history['val_accuracy']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'r')
    plt.plot(epochs, vacc, 'b')
    #plt.title('Training and validation accuracy')
    plt.ylabel('Accuracy %')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='lower right')
    plt.grid(True)
    plt.savefig('{}/{}_acc.jpg'.format(LOG_PATH, EXPERIMENT_NAME), dpi=100)
    plt.close('all')
    #plt.show()

plot_loss(model.history)
plot_acc(model.history)
print("Done training and logging!")


# In[9]:


from keras.models import load_model

model = None
model = load_model("{}/{}.h5".format(LOG_PATH, EXPERIMENT_NAME), compile = False)
model.summary()


# In[11]:


# Make predictions using trained model
y_pred = model.predict(x_test, verbose=1)
print("Predictions: ", y_pred.shape)

# Convert ground truth to column values
y_test_flat = np.argmax(y_test, axis=1)
print("After flattening ground truth: ", y_test_flat.shape)


# Get labels from predictions
y_pred_flat = np.array([np.argmax(pred) for pred in y_pred]) 
print("Binarize probability values: ", y_pred_flat.shape)

assert y_pred_flat.shape == y_test_flat.shape, "Shape mismatch!"


# In[12]:


# Sanity check

print(y_test.shape, y_test_flat.shape, y_pred.shape, y_pred_flat.shape)


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
print('Area under ROC curve : ', roc_auc_score(y_test, y_pred) *100 )


# In[21]:


from sklearn.metrics import f1_score
F1_score = f1_score(y_test_flat, y_pred_flat, average='weighted')
print("F1 score: ", F1_score)

print("------------------------------------End of script------------------------------------")




