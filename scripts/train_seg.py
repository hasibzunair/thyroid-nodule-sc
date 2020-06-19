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
from keras.utils import generic_utils
from keras import callbacks
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


# Go back one step to read module
import sys
sys.path.insert(0,"..") 

import data_utils
import classes
import models as M
import losses as l



# %%
# Name data and config types
DATASET_NAME = "data0" # name of the npz file
SRUNET_DATA = "data0_unet_data_augment" # SRUNET data path
CFG_NAME = "unet_data_augment" # name of the architecture/configuration for segmentation model


epoch_list = [15,25,35,45]
unet_or_srunet = 0 #0 for Unet, 1 for SRNET

ROOT_DIR = os.path.abspath("../")
DATASET_FOLDER = "npy_data"
DATASET_PATH = os.path.join(ROOT_DIR, "datasets", DATASET_FOLDER)
SRUNET_DATA_PATH = os.path.join(ROOT_DIR, "logs", SRUNET_DATA, "sr_unetdata")
EXPERIMENT_NAME = "{}_{}".format(DATASET_NAME, CFG_NAME)


# %%
# Train
batch_size = 32
epochs = 100000
interval = 10 #show correct dice and log it after every ? epochs

if not os.path.exists(os.path.join(ROOT_DIR, "logs")):
    os.mkdir(os.path.join(ROOT_DIR, "logs"))

# Make log path to store all results
LOG_PATH = os.path.join(ROOT_DIR, "logs", EXPERIMENT_NAME)

if not os.path.exists(LOG_PATH):
    os.mkdir(LOG_PATH)
    
print(os.listdir(DATASET_PATH))


# Load the dataset
if unet_or_srunet == 0:
    data = np.load(DATASET_PATH + '/{}.npz'.format(DATASET_NAME))
    train_data = data['name1']
    train_labels = data['name2']

    train_data = np.expand_dims(train_data, axis=-1)
    train_labels = np.minimum(train_labels, 1)
    train_labels = np.expand_dims(train_labels, axis=-1)
    print(train_data.shape, train_labels.shape)

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


else: #for SRNET
    for indx in range(len(epoch_list)):

        data = np.load(SRUNET_DATA_PATH + '/train_pred_%s.npz'%(epoch_list[indx]))

        train_data = (data['name1'])
        train_labels = (data['name2'])

        split = 2332 #80% of the data as training and rest as validation. This is different than unet because we are only using training set for SRUNET

        if indx == 0:
            x_train = train_data[:split] #Splitting is done per epoch output to make sure cases from validation are not seen during training
            x_test = train_data[split:]
            y_train = train_labels[:split]
            y_test = train_labels[split:]
        else:
            x_train = np.concatenate((x_train, train_data[:split]))
            x_test = np.concatenate((x_test, train_data[split:]))

            y_train = np.concatenate((y_train, train_labels[:split]))
            y_test = np.concatenate((y_test, train_labels[split:]))

        del data, train_data, train_labels


    print("Train and validate on -------> ", x_train.shape, x_test.shape, y_train.shape, y_test.shape)




# %%
# Plot and save accuravy loss graphs individually
def plot_loss_accu(history):
    loss = history.history['loss'][1:]
    val_loss = history.history['val_loss'][1:]
    epochs = range(len(loss))
    plt.plot(epochs, loss, 'g')
    plt.plot(epochs, val_loss, 'y')
    plt.title('Training and validation loss')
    plt.ylabel('Loss %')
    plt.xlabel('Epoch')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.grid(True)
    plt.savefig('{}/{}_loss.png'.format(LOG_PATH, EXPERIMENT_NAME), dpi=100)
    #plt.show()
    
    loss = history.history['jacard'][1:]
    val_loss = history.history['val_jacard'][1:]
    epochs = range(len(loss))
    plt.plot(epochs, loss, 'r')
    plt.plot(epochs, val_loss, 'b')
    plt.title('Training and validation jaccard index')
    plt.ylabel('Jaccard Index %')
    #plt.xlabel('Epoch')
    plt.legend(['training', 'validation'], loc='lower right')
    plt.grid(True)
    plt.savefig('{}/{}_jac.png'.format(LOG_PATH, EXPERIMENT_NAME), dpi=100)
    #plt.show()
    
    loss = history.history['dice'][1:]
    val_loss = history.history['val_dice'][1:]
    epochs = range(len(loss))
    plt.plot(epochs, loss, 'r')
    plt.plot(epochs, val_loss, 'b')
    plt.title('Training and validation dice')
    plt.ylabel('Dice Score %')
    #plt.xlabel('Epoch')
    plt.legend(['training', 'validation'], loc='lower right')
    plt.grid(True)
    plt.savefig('{}/{}_jac.png'.format(LOG_PATH, EXPERIMENT_NAME), dpi=100)
    #plt.show()
    
    
def plot_graphs(history):
    
    # b, g, r, y, o, -g, -m
    
    experiment_name = EXPERIMENT_NAME
    plt.figure(figsize=(18, 5))
    plt.subplot(131)
    plt.plot(history.history['loss'],linewidth=4)
    plt.plot(history.history['val_loss'],linewidth=4)
    plt.title('{} loss'.format(experiment_name))
    #plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid(True)

    

    plt.subplot(132)
    plt.plot(history.history['jacard'],linewidth=4)
    plt.plot(history.history['val_jacard'],linewidth=4)
    plt.title('{} Jacard'.format(experiment_name))
    #plt.ylabel('Jacard')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.legend(['Train', 'Test'], loc='upper left')



    plt.subplot(133)
    plt.plot(history.history['dice_coef'],linewidth=4)
    plt.plot(history.history['val_dice_coef'],linewidth=4)
    plt.title('{} Dice'.format(experiment_name))
    #plt.ylabel('Dice')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid(True)
    plt.savefig('{}/{}_graph.png'.format(LOG_PATH, EXPERIMENT_NAME), dpi=100)
    #plt.show()

# %%
# Build standard U-Net model
if unet_or_srunet == 0:
    print("Segmentation model")
    model = M.unet(input_size = (train_data.shape[1], train_data.shape[2], train_data.shape[-1]))
else:
    print("Shape regularization model")
    model = M.SRUNET(input_size = (x_train.shape[1], x_train.shape[2], x_train.shape[-1]))
# Build U-Net model with custom encoder
#backbone_name = 'vgg16'
#encoder_weights = None
#model = M.unet_backbone(backbone=backbone_name, input_size = (train_data.shape[1], 
#            train_data.shape[2], train_data.shape[-1]), encoder_weights=encoder_weights)

model.summary()

# %%
"""
## Callbacks
"""

# %%
# Callbacks
weights_path = "{}/{}.h5".format(LOG_PATH, EXPERIMENT_NAME)
checkpointer = ModelCheckpoint(filepath=weights_path, verbose=1, monitor='val_jacard', mode='max', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_jacard', factor=0.1, patience=5, verbose=1, min_lr=1e-8, mode='max') # new_lr = lr * factor
early_stopping = EarlyStopping(monitor='val_jacard', min_delta=0, verbose=1, patience=8, mode='max', restore_best_weights=True)
csv_logger = CSVLogger('{}/{}_training.csv'.format(LOG_PATH, EXPERIMENT_NAME))

ie = classes.IntervalEvaluation(EXPERIMENT_NAME, LOG_PATH, interval,unet_or_srunet, validation_data=(x_test, y_test),
                                training_data=(x_train, y_train))




#generators
training_generator = classes.DataGenerator_Augment(x_train, y_train, batch_size=batch_size, shuffle=True)
validation_generator = classes.DataGenerator_Augment(x_test, y_test, batch_size=batch_size, shuffle=True)


start_time = time.time()


# Train model on dataset
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,callbacks=[ie, checkpointer, reduce_lr, csv_logger, early_stopping],
                shuffle=True,
                verbose = 2,epochs = epochs, steps_per_epoch= (len(x_train)*2) // batch_size)





end_time = time.time()
print("--- Time taken to train : %s hours ---" % ((end_time - start_time)//3600))



# %%
# Log training history
plot_graphs(model.history)

# %%
# Evaluate trained model using Jaccard and Dice metric
from tensorflow.keras.models import load_model
model = None
model = load_model(weights_path, compile=False)
yp = model.predict(x=x_test, batch_size=16, verbose=1)
#Round off boolean masks
yp = np.round(yp,0)
yp.shape

# %%
y_test.shape, yp.shape

# %%
# Eval on train set
yp_t = None
yp_t = model.predict(x=x_train, batch_size=16, verbose=0)
#Round off boolean masks
yp_t = np.round(yp_t,0)
yp_t.shape

# %%
yp_t.shape, yp.shape

# %%
y_train.shape, y_test.shape

# %%
np.savez('{}/{}_{}_mask_pred.npz'.format(LOG_PATH, CFG_NAME, DATASET_NAME), 
         name1=y_train, name2=y_test, name3=yp_t, name4=yp)

# %%
# binary segmentation

# try:
#     os.makedirs('{}/results/'.format(LOG_PATH, EXPERIMENT_NAME))
# except:
#     pass 

# for i in range(5):
    
#     plt.figure(figsize=(20,10))
#     plt.subplot(1,3,1)
#     if len(x_test[i].shape) >= 2:
#         plt.grid(False)
#         plt.imshow(x_test[i].squeeze(), cmap='gray') # 1-channel image
#     else:
#         plt.grid(False)
#         plt.imshow(x_test[i]) # 3-channel
        
#     plt.title('Input')
#     plt.subplot(1,3,2)
#     plt.grid(False)
#     plt.imshow(y_test[i].reshape(y_test[i].shape[0],y_test[i].shape[1]), cmap='magma') #cmap='magma'
#     plt.title('Ground Truth')
#     plt.subplot(1,3,3)
#     plt.grid(False)
#     plt.imshow(yp[i].reshape(yp[i].shape[0],yp[i].shape[1]),cmap='magma')
#     plt.title('Prediction')
    
#     # Calc jaccard index of predictions
#     intersection = yp[i].ravel() * y_test[i].ravel()
#     union = yp[i].ravel() + y_test[i].ravel() - intersection
#     jacard = (np.sum(intersection)/np.sum(union))  
    
#     plt.suptitle('Jacard Index: '+ str(np.sum(intersection)) +'/'+ str(np.sum(union)) +' = '+str(jacard))
#     plt.savefig('{}/results/'.format(LOG_PATH, EXPERIMENT_NAME)+str(i)+'.png',format='png')
#     plt.show()
#     plt.close()

# %%