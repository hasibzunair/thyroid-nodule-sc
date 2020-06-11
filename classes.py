# from Plot_Utils import *
from keras.models import *
from keras.layers import *
import numpy as np
import keras
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from keras import models

import math
import glob
import time
from keras import backend as K

from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.callbacks import Callback
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
import numpy as np
# from datetime import datetime
import tensorflow as tf
# from sklearn.metrics import roc_curve, auc


from keras.optimizers import *
import os

from tensorflow import keras
import data_utils
import models
import metrics
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import pandas

import logging as l



class IntervalEvaluation(Callback):
	def __init__(self, model_name, logging_dir, interval, validation_data=(), training_data=()):
		super(Callback, self).__init__()
		self.interval = interval
		self.model_name = model_name
		self.logging_dir = logging_dir

		self.X_val, self.y_val = validation_data
		self.X_train, self.y_train = training_data

	def on_train_begin(self, logs={}):
		self.train_accuracy = []
		self.train_sensitivity = []
		self.train_specificity = []
		self.train_dice = []

		self.val_accuracy = []
		self.val_sensitivity = []
		self.val_specificity = []
		self.val_dice = []

		self.epochz = []

	def on_train_end(self, logs={}):
		return

	def on_epoch_begin(self, epoch, logs={}):
		return

	def on_epoch_end(self, epoch, logs={}):

		y_pred_val = self.model.predict(self.X_val, verbose=0)
		y_pred_train = self.model.predict(self.X_train, verbose=0)

		# For finding AUC
		# fpr, tpr, _ = roc_curve(self.y_train.flatten()>0.5, y_pred_train.flatten())
		# roc_auc_train = auc(fpr, tpr)
		#
		# fpr, tpr, _ = roc_curve(self.y_val.flatten()>0.5, y_pred_val.flatten())
		# roc_auc_val = auc(fpr, tpr)

		operation_point, _, _, accuracy, specificity, sensitivity, dice = data_utils.get_operating_points(
			self.y_train.flatten(), y_pred_train.flatten())

		print(
			"\nTraining Operating Point:{:.4f}, Accuracy :{:.4f}, Sensitivity:{:.4f}, Specificity: {:.4f}, Dice: {:.4f}".format(
				operation_point, accuracy, sensitivity, specificity, dice))

		# self.train_operating_point.append(operation_point)
		# self.train_aucs.append(roc_auc_train)
		self.epochz.append(epoch)

		self.train_accuracy.append(accuracy)
		self.train_sensitivity.append(sensitivity)
		self.train_specificity.append(specificity)
		self.train_dice.append(dice)

		operation_point, _, _, accuracy, specificity, sensitivity, dice = data_utils.use_operating_points(
			operation_point,
			self.y_val.flatten(), y_pred_val.flatten())

		print(
			"Validation Operating Point:{:.4f}, Accuracy :{:.4f}, Sensitivity:{:.4f}, Specificity: {:.4f}, Dice: {:.4f}\n".format(
				operation_point, accuracy, sensitivity, specificity, dice))

		# Can be used later for drawing plots
		# self.val_operating_point.append(operation_point)
		# self.val_aucs.append(roc_auc_val)
		self.val_accuracy.append(accuracy)
		self.val_sensitivity.append(sensitivity)
		self.val_specificity.append(specificity)
		self.val_dice.append(dice)

		# Save doing subplot
		figure_path = os.path.join(self.logging_dir, 'figures')
		if not os.path.isdir(figure_path):
			os.mkdir(figure_path)

		# save 5 training and 5 Validation at each epoch
		data_utils.saveResultasPlot(figure_path, epoch, self.X_train, self.y_train, y_pred_train, 'Training', 5)
		data_utils.saveResultasPlot(figure_path, epoch, self.X_val, self.y_val, y_pred_val, 'Validation', 5)

		# if epoch % self.interval == 0:
		#     np.savez(os.path.join(self.logging_dir, self.model_name, 'val_metrics'),
		#              name1=self.val_accuracy,
		#              name2=self.val_sensitivity,
		#              name3=self.val_specificity, name4=self.val_dice)
		#
		#     np.savez(os.path.join(self.logging_dir, self.model_name,'train_metrics'),
		#              name1=self.train_accuracy,
		#              name2=self.train_sensitivity,
		#              name3=self.train_specificity, name4=self.train_dice)

		# save as csv file
		df = pandas.DataFrame(data={"epoch": self.epochz, "train_accuracy": self.train_accuracy,
									"train_sensitivity": self.train_sensitivity,
									"train_specificity": self.train_specificity, "train_dice": self.train_dice,
									"val_accuracy": self.val_accuracy, "val_sensitivity": self.val_sensitivity,
									"val_specificity": self.val_specificity, "val_dice": self.val_dice})
		df.to_csv(os.path.join(self.logging_dir, 'log2.csv'), sep=',', index=False)

		# This part is for generating prediction from intermediate epochs
		save_at_epochs = [15, 25, 35, 45, 55]

		if (np.sum((epoch) == np.transpose(save_at_epochs)) > 0):

			print('\nReached Epoch Number %s and saving intermediate epoch results...' % (
				epoch))

			# Save doing subplot
			data_path = os.path.join(self.logging_dir, 'sr_unetdata')
			if not os.path.isdir(data_path):
				os.mkdir(data_path)

			np.savez(os.path.join(data_path, 'train_pred_' + str(epoch)),
					 name1=y_pred_train,
					 name2=self.y_train)

			self.model.save(data_path + '/model_' + str(epoch) + '.h5')

			# sanity check plotting
			data_path = os.path.join(data_path, 'figures')
			if not os.path.isdir(data_path):
				os.mkdir(data_path)

			data_utils.saveCorruptionResults(data_path, epoch, y_pred_train, self.y_train, 'Train', 5)



class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, Xdata, Ydata, batch_size=32, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.Xdata = Xdata
        self.Ydata = Ydata
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.Xdata) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.Xdata))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Generate data
        X = self.Xdata[indexes]
        y = self.Ydata[indexes]


        # #Corruption part of the code
        # percentage_corruption = 0.30
		#
        # for i in range(len(y)):
		#
        #     img = y[i]
		#
        #     #Generating
		#
		#
        #     k1 = np.random.rand(np.shape(img)[0], np.shape(img)[1], np.shape(img)[2])
        #     k2 = np.random.rand(np.shape(img)[0], np.shape(img)[1], np.shape(img)[2])
        #     k3 = np.random.rand(np.shape(img)[0], np.shape(img)[1], np.shape(img)[2])
        #     k4 = np.random.rand(np.shape(img)[0], np.shape(img)[1], np.shape(img)[2])
		#
        #     mask_indx = np.asarray(np.where(img==1))
        #     mask_indx = np.transpose(mask_indx)
        #     np.random.shuffle(mask_indx)
		#
        #     mask_indx = mask_indx[:int(np.floor(len(mask_indx)*percentage_corruption)),:]
        #     mask_indx = np.transpose(mask_indx)
		#
        #     mask = np.zeros(np.shape(img))
        #     mask[mask_indx[0], mask_indx[1], mask_indx[2]] = 1
		#
        #     new = ((img*mask*k1) + (img*mask*k2) + (img*mask*k3) +(img*mask*k4))/4





        return X, y