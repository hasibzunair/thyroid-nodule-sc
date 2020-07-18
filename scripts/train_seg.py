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

# Go back one step to read module
import sys
sys.path.insert(0,"..") 

import segmentation_models as sm
import data_utils
import classes
import models as M
import losses as l
import metrics


# %%
# Name data and config types
DATASET_NAME = "data0" # name of the npz file
SRUNET_DATA = "data0_unet_data_augment" # SRUNET data path
CFG_NAME = "Cascaded_efb0_no_augmentation" # name of the architecture/configuration for segmentation model
TRAINED_SRNET = "data0_data0_SRNET_with_augmented_data_[6, 10, 12, 16, 20]" # Path of SR-Unet weight 

epoch_list = [10, 12, 16, 20]
unet_or_srunet = 2 #0 for Unet, 1 for SRNET, #2 cascaded
save_inter_layers_flag = 0 #this is used for saving intermediate layer results (set to ) if training SRUNET is not required)

augmentation_flag = 0
## This part is to load best predictions as validation set for SRUNET and used all training set for training
#previously 20% of training was used for validation
load_predictions_from_best_model = 1
best_model = 'data0_unet_efb0'


# Configs for custom encoder
encoder_flag = 1 # Set 1 to use custom encoder in Unet
backbone_name = 'efficientnetb3'
encoder_weights = "imagenet"
    
    
ROOT_DIR = os.path.abspath("../")
DATASET_FOLDER = "npy_data"
#DATASET_FOLDER = "/home/hasib/scratch/npy_data" # use this when on server
DATASET_PATH = os.path.join(ROOT_DIR, "datasets", DATASET_FOLDER)
SRUNET_DATA_PATH = os.path.join(ROOT_DIR, "logs", SRUNET_DATA, "sr_unetdata")

if unet_or_srunet == 1:
    EXPERIMENT_NAME = "{}_{}_{}".format(DATASET_NAME, CFG_NAME, epoch_list)
else:
    EXPERIMENT_NAME = "{}_{}".format(DATASET_NAME, CFG_NAME)

TRAINED_SRUNET_PATH = os.path.join(ROOT_DIR, "logs", TRAINED_SRNET)

# %%
# Train
lr = 0.0001 # 0.0001
batch_size = 8
epochs = 300
interval = 10 #10 #show correct dice and log it after every ? epochs
optim = 'adam' #keras.optimizers.Adam(lr)
loss_func = 'binary_crossentropy'#l.dice_coef_loss


# Define custom loss
################################################

dice_loss = sm.losses.DiceLoss()
jacard_loss = sm.losses.JaccardLoss()
focal_loss = sm.losses.BinaryFocalLoss()

total_loss = (0.5 * focal_loss) + (1 * jacard_loss)

# Comment this line to not use custom loss, use what is defined at the top
loss_func = focal_loss
################################################


if not os.path.exists(os.path.join(ROOT_DIR, "logs")):
    os.mkdir(os.path.join(ROOT_DIR, "logs"))

# Make log path to store all results
LOG_PATH = os.path.join(ROOT_DIR, "logs", EXPERIMENT_NAME)

if not os.path.exists(LOG_PATH):
    os.mkdir(LOG_PATH)
    
print(os.listdir(DATASET_PATH))


# Load the dataset
if (unet_or_srunet == 0 or unet_or_srunet ==2):
    data = np.load(DATASET_PATH + '/{}.npz'.format(DATASET_NAME))
    train_data = data['name1']/255
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




elif unet_or_srunet == 1: #for SRNET

    #This part loads the same examples used as validaiton and perfroms prediction using the best trained UNET,
    #this is then used for validating the use of SRUNET.
    if load_predictions_from_best_model == 1:
        data = np.load(DATASET_PATH + '/{}.npz'.format(DATASET_NAME))
        train_data = data['name1']
        train_labels = data['name2']
        train_data = np.expand_dims(train_data, axis=-1)
        train_labels = np.minimum(train_labels, 1)
        train_labels = np.expand_dims(train_labels, axis=-1)

        x_test = train_data[2915:]
        y_test = train_labels[2915:]

        weights_path = "../logs/{}/{}.h5".format(best_model, best_model)
        model = None
        model = load_model(weights_path, compile=False)
        x_test = model.predict(x=x_test, batch_size=16, verbose=1)




    for indx in range(len(epoch_list)):

        data = np.load(SRUNET_DATA_PATH + '/train_pred_%s.npz'%(epoch_list[indx]))

        train_data = (data['name1'])
        train_labels = (data['name2'])

        if load_predictions_from_best_model == 0:
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

        #All data used as training and validation is basically the prediction from best UNET
        else:
            if indx == 0:
                x_train = train_data
                y_train = train_labels
            else:
                x_train = np.concatenate((x_train, train_data))
                y_train = np.concatenate((y_train, train_labels))

            del data, train_data, train_labels

print("Train and validate on -------> ", x_train.shape, x_test.shape, y_train.shape, y_test.shape)

print("\n\nX Train- max: %s, min: %s" %(np.max(x_train), np.min(x_train)))
print("Y Train- max: %s, min: %s" % (np.max(y_train), np.min(y_train)))
print("X Val- max: %s, min: %s" % (np.max(x_test), np.min(x_test)))
print("Y Val- max: %s, min: %s" % (np.max(y_test), np.min(y_test)))



# %%
# Build standard U-Net model
if (unet_or_srunet == 0 and encoder_flag == 0):
    print("Segmentation model")
    # Vanilla U-Net
    model = M.unet(input_size = (train_data.shape[1], train_data.shape[2], train_data.shape[-1]))
    # Compiling
    model.compile(optimizer=optim, loss=loss_func, metrics=[metrics.jacard, metrics.dice_coef])

if (unet_or_srunet == 1 and encoder_flag == 0):
    print("Shape regularization model")
    model = M.SRUNET(input_size = (x_train.shape[1], x_train.shape[2], x_train.shape[-1]))

    # Compiling
    model.compile(optimizer=optim, loss=loss_func, metrics=[metrics.jacard, metrics.dice_coef])


if unet_or_srunet == 2:
    print("Cascaded Network")

    if encoder_flag == 1:
        # Build U-Net model with custom encoder
        unet_main = M.unet_backbone(backbone=backbone_name, input_size = (train_data.shape[1],
                    train_data.shape[2], train_data.shape[-1]), encoder_weights=encoder_weights)

    else:
        unet_main = M.unet(input_size=(train_data.shape[1], train_data.shape[2], train_data.shape[-1]))

    srunet = M.SRUNET_cascade(input_size = (x_train.shape[1], x_train.shape[2], x_train.shape[-1]))
    srunet.load_weights(TRAINED_SRUNET_PATH + '/' + TRAINED_SRNET + '.h5')

    #freezing pretrained SRUNET
    srunet.trainable = False

    # Defining Cascaded Architechture
    encoded_gt = Input(shape=(16,16,512)) #shape of encoded output

    inputs = Input(shape=(train_data.shape[1], train_data.shape[2], train_data.shape[-1]))
    unet_output = unet_main(inputs)
    encoded, output = srunet(unet_output)
    model = Model(inputs=[inputs, encoded_gt], outputs=[output])

    optim = 'adam'

    #loss_func = abs(unet -out)**2 +a(encodere(GT)-encoder(unet)) + b(abs(GT-unet)**2)
    # Define custom loss
    bce = tf.keras.losses.BinaryCrossentropy()
    def custom_loss(unet_output, encoded, encoded_gt):


        def loss(y_true, y_pred):
            a = 0.5
            b = 0.5


            loss = tf.keras.backend.sqrt(metrics.mas(unet_output, y_pred) + a * (metrics.mas(encoded_gt, encoded)) + b * (
                metrics.mas(y_true, unet_output)))

            # loss = bce(y_true, y_pred) + (
            #     20*bce(y_true, unet_output))

            return loss

        return loss


    model.compile(optimizer=optim, loss = custom_loss(unet_output, encoded, encoded_gt), metrics=[metrics.jacard, metrics.dice_coef])
    
if unet_or_srunet == 0 and encoder_flag == 1:
    
    print("Unet with custom encoder")
    # Build U-Net model with custom encoder
    model = M.unet_backbone(backbone=backbone_name, input_size = (train_data.shape[1], 
                train_data.shape[2], train_data.shape[-1]), encoder_weights=encoder_weights)

    # Compiling
    model.compile(optimizer=optim, loss=loss_func, metrics=[metrics.jacard, metrics.dice_coef])


if (unet_or_srunet == 1 and encoder_flag == 1):
    print("Shape regularization model")
    model = M.SRUNET_backbone(backbone=backbone_name, input_size = ((x_train.shape[1], x_train.shape[2], x_train.shape[-1])), encoder_weights=encoder_weights)

    # Compiling
    model.compile(optimizer=optim, loss=loss_func, metrics=[metrics.jacard, metrics.dice_coef])


    
model.summary()



start_time = time.time()


if (unet_or_srunet ==0 or unet_or_srunet == 1):

    print("Train Unet with data augmentation.")
    # Callbacks
    weights_path = "{}/{}.h5".format(LOG_PATH, EXPERIMENT_NAME)
    checkpointer = ModelCheckpoint(filepath=weights_path, verbose=1, monitor='val_jacard', mode='max',
                                   save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_jacard', factor=0.1, patience=5, verbose=1, min_lr=1e-8,
                                  mode='max')  # new_lr = lr * factor
    early_stopping = EarlyStopping(monitor='val_jacard', min_delta=0, verbose=1, patience=18, mode='max',
                                   restore_best_weights=True)
    csv_logger = CSVLogger('{}/{}_training.csv'.format(LOG_PATH, EXPERIMENT_NAME))

    ie = classes.IntervalEvaluation(save_inter_layers_flag, EXPERIMENT_NAME, LOG_PATH, interval, unet_or_srunet,
                                    validation_data=(x_test, y_test),
                                    training_data=(x_train, y_train))

    if augmentation_flag == 1:
        #generators
        training_generator = classes.DataGenerator_Augment(x_train, y_train, batch_size=batch_size, shuffle=True)
        validation_generator = classes.DataGenerator_Augment(x_test, y_test, batch_size=batch_size, shuffle=True)

        #Train model on dataset
        model.fit_generator(generator=training_generator,
                            validation_data=validation_generator,callbacks=[ie, checkpointer, reduce_lr, csv_logger, early_stopping],
                        shuffle=True,
                        verbose = 2,epochs = epochs, steps_per_epoch= (len(x_train)*2) // batch_size)

    else:
        model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_test,y_test),
                        callbacks=[ie, checkpointer, reduce_lr, csv_logger, early_stopping],
                        shuffle=True,
                        verbose = 2)




    # %%
    # Log training history
    data_utils.plot_graphs(model.history,LOG_PATH, EXPERIMENT_NAME)

elif (unet_or_srunet ==2):
    #[unet_output, encoded, output]
    #Generating  Encoded results of GT in advance, with has to be inserted to the generators if we wish to use augmentation later
    y_train_encoded, _ = srunet.predict(x=y_train, batch_size=16, verbose=2)
    y_test_encoded, _ = srunet.predict(x=y_test, batch_size=16, verbose=2)


    # #generators
    # training_generator = classes.DataGenerator_Augment(x_train, y_train, batch_size=batch_size, shuffle=True)
    # validation_generator = classes.DataGenerator_Augment(x_test, y_test, batch_size=batch_size, shuffle=True)

    # %%
    # Callbacks
    weights_path = "{}/{}.h5".format(LOG_PATH, EXPERIMENT_NAME)
    checkpointer = ModelCheckpoint(filepath=weights_path, verbose=1, monitor='val_jacard', mode='max',
                                   save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_jacard', factor=0.1, patience=5, verbose=1, min_lr=1e-8,
                                  mode='max')  # new_lr = lr * factor
    early_stopping = EarlyStopping(monitor='val_jacard', min_delta=0, verbose=1, patience=20, mode='max',
                                   restore_best_weights=True)
    csv_logger = CSVLogger('{}/{}_training.csv'.format(LOG_PATH, EXPERIMENT_NAME))
    ie = classes.IntervalEvaluation_cascaded(EXPERIMENT_NAME, LOG_PATH, interval, unet_or_srunet,unet_main,
                                    validation_data=(x_test, y_test_encoded, y_test),
                                    training_data=(x_train, y_train_encoded, y_train))

    if augmentation_flag == 1:
        #generators
        training_generator = classes.DataGenerator_Augment_cascaded(x_train, srunet, y_train, batch_size=batch_size, shuffle=True)
        validation_generator = classes.DataGenerator_Augment_cascaded(x_test, srunet, y_test, batch_size=batch_size, shuffle=True)

        #Train model on dataset
        model.fit_generator(generator=training_generator,
                            validation_data=validation_generator,callbacks=[checkpointer, reduce_lr, csv_logger, early_stopping],
                        shuffle=True,
                        verbose = 2,epochs = epochs, steps_per_epoch= (len(x_train)*2) // batch_size)

    else:
        model.fit([x_train, y_train_encoded], y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=([x_test, y_test_encoded],y_test),
                        callbacks=[checkpointer, reduce_lr, csv_logger, early_stopping],
                        shuffle=True,
                        verbose = 2)

    # %%
    # Log training history
    data_utils.plot_graphs(model.history,LOG_PATH, EXPERIMENT_NAME)


end_time = time.time()
print("--- Time taken to train : %s hours ---" % ((end_time - start_time)//3600))





# %%
# Evaluate trained model using Jaccard and Dice metric

# model = None
# model = load_model(weights_path, compile=False)
# yp = model.predict(x=x_test, batch_size=16, verbose=1)
# #Round off boolean masks
# yp = np.round(yp,0)
# yp.shape
#
# # %%
# y_test.shape, yp.shape
#
# # %%
# # Eval on train set
# yp_t = None
# yp_t = model.predict(x=x_train, batch_size=16, verbose=0)
# #Round off boolean masks
# yp_t = np.round(yp_t,0)
# yp_t.shape
#
# # %%
# yp_t.shape, yp.shape
#
# # %%
# y_train.shape, y_test.shape
#
# # %%
# np.savez('{}/{}_{}_mask_pred.npz'.format(LOG_PATH, CFG_NAME, DATASET_NAME),
#          name1=y_train, name2=y_test, name3=yp_t, name4=yp)

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
