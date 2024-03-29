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
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from imgaug import augmenters as iaa #Install imgaug library (for data augmentation) from https://github.com/aleju/imgaug#installation
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
	def __init__(self, save_inter_layers_flag, model_name, logging_dir, interval, unet_or_srunet,validation_data=(), training_data=()):
		super(Callback, self).__init__()
		self.interval = interval
		self.model_name = model_name
		self.logging_dir = logging_dir
		self.unet_or_srunet = unet_or_srunet
		self.save_inter_layers_flag = save_inter_layers_flag

		self.X_val, self.y_val = validation_data
		self.X_train, self.y_train = training_data

		#for speeding up evaluation
		randomize = np.arange(len(self.X_train))
		np.random.shuffle(randomize)
		self.X_train = self.X_train[randomize[:300]]
		self.y_train = self.y_train[randomize[:300]]


	def on_train_begin(self, logs={}):
		self.train_accuracy_in = []
		self.train_sensitivity_in = []
		self.train_specificity_in = []
		self.train_dice_in = []
		self.train_jaccard_in = []

		self.train_accuracy = []
		self.train_sensitivity = []
		self.train_specificity = []
		self.train_dice = []
		self.train_jaccard = []

		self.val_accuracy_in = []
		self.val_sensitivity_in = []
		self.val_specificity_in = []
		self.val_dice_in = []
		self.val_jaccard_in = []

		self.val_accuracy = []
		self.val_sensitivity = []
		self.val_specificity = []
		self.val_dice = []
		self.val_jaccard = []

		self.epochz = []

	def on_train_end(self, logs={}):
		return

	def on_epoch_begin(self, epoch, logs={}):
		return

	def on_epoch_end(self, epoch, logs={}):

		y_pred_val = self.model.predict(self.X_val, verbose=0)
		y_pred_train = self.model.predict(self.X_train, verbose=0)


		# SAving predictions per epoch
		figure_path = os.path.join(self.logging_dir, 'figures')
		if not os.path.isdir(figure_path):
			os.mkdir(figure_path)

		# save 5 training and 5 Validation at each epoch
		data_utils.saveResultasPlot(self.unet_or_srunet, figure_path, epoch, self.X_train, self.y_train, y_pred_train, 'Training', 5)
		data_utils.saveResultasPlot(self.unet_or_srunet, figure_path, epoch, self.X_val, self.y_val, y_pred_val, 'Validation', 5)

		if (self.unet_or_srunet == 0):
			if epoch % self.interval == 0:
				# For finding AUC
				# fpr, tpr, _ = roc_curve(self.y_train.flatten()>0.5, y_pred_train.flatten())
				# roc_auc_train = auc(fpr, tpr)
				#
				# fpr, tpr, _ = roc_curve(self.y_val.flatten()>0.5, y_pred_val.flatten())
				# roc_auc_val = auc(fpr, tpr)

				operation_point, _, _, accuracy, specificity, sensitivity, dice, jaccard = data_utils.get_operating_points(
					self.y_train.flatten(), y_pred_train.flatten())

				print(
					"\nTraining Operating Point:{:.4f}, Accuracy :{:.4f}, Sensitivity:{:.4f}, Specificity: {:.4f}, Dice: {:.4f}, Jaccard: {:.4f}".format(
						operation_point, accuracy, sensitivity, specificity, dice, jaccard))

				# self.train_operating_point.append(operation_point)
				# self.train_aucs.append(roc_auc_train)
				self.epochz.append(epoch)

				self.train_accuracy.append(accuracy)
				self.train_sensitivity.append(sensitivity)
				self.train_specificity.append(specificity)
				self.train_dice.append(dice)
				self.train_jaccard.append(jaccard)

				operation_point, _, _, accuracy, specificity, sensitivity, dice, jaccard = data_utils.use_operating_points(
					operation_point,
					self.y_val.flatten(), y_pred_val.flatten())

				print(
					"Validation Operating Point:{:.4f}, Accuracy :{:.4f}, Sensitivity:{:.4f}, Specificity: {:.4f}, Dice: {:.4f}, Jaccard: {:.4f}\n".format(
						operation_point, accuracy, sensitivity, specificity, dice, jaccard))

				# Can be used later for drawing plots
				# self.val_operating_point.append(operation_point)
				# self.val_aucs.append(roc_auc_val)
				self.val_accuracy.append(accuracy)
				self.val_sensitivity.append(sensitivity)
				self.val_specificity.append(specificity)
				self.val_dice.append(dice)
				self.val_jaccard.append(jaccard)


				# np.savez(os.path.join(self.logging_dir, self.model_name, 'val_metrics'),
				#          name1=self.val_accuracy,
				#          name2=self.val_sensitivity,
				#          name3=self.val_specificity, name4=self.val_dice)
				#
				# np.savez(os.path.join(self.logging_dir, self.model_name,'train_metrics'),
				#          name1=self.train_accuracy,
				#          name2=self.train_sensitivity,
				#          name3=self.train_specificity, name4=self.train_dice)

				# save as csv file
				df = pandas.DataFrame(data={"epoch": self.epochz, "train_accuracy": self.train_accuracy,
											"train_sensitivity": self.train_sensitivity,
											"train_specificity": self.train_specificity, "train_dice": self.train_dice,"train_jaccard": self.train_jaccard,
											"val_accuracy": self.val_accuracy, "val_sensitivity": self.val_sensitivity,
											"val_specificity": self.val_specificity, "val_dice": self.val_dice, "val_jaccard": self.val_jaccard})
				df.to_csv(os.path.join(self.logging_dir, 'log2.csv'), sep=',', index=False)
		if (self.unet_or_srunet == 1):
			if epoch % self.interval == 0:
				# For finding AUC
				# fpr, tpr, _ = roc_curve(self.y_train.flatten()>0.5, y_pred_train.flatten())
				# roc_auc_train = auc(fpr, tpr)
				#
				# fpr, tpr, _ = roc_curve(self.y_val.flatten()>0.5, y_pred_val.flatten())
				# roc_auc_val = auc(fpr, tpr)

				operation_point, _, _, accuracy, specificity, sensitivity, dice, jaccard = data_utils.get_operating_points(
					self.y_train.flatten(), self.X_train.flatten())

				print(
					"\nIn: Training Operating Point:{:.4f}, Accuracy :{:.4f}, Sensitivity:{:.4f}, Specificity: {:.4f}, Dice: {:.4f}, Jaccard: {:.4f}".format(
						operation_point, accuracy, sensitivity, specificity, dice, jaccard))

				self.train_accuracy_in.append(accuracy)
				self.train_sensitivity_in.append(sensitivity)
				self.train_specificity_in.append(specificity)
				self.train_dice_in.append(dice)
				self.train_jaccard_in.append(jaccard)


				operation_point, _, _, accuracy, specificity, sensitivity, dice, jaccard = data_utils.get_operating_points(
					self.y_train.flatten(), y_pred_train.flatten())


				print(
					"Out: Training Operating Point:{:.4f}, Accuracy :{:.4f}, Sensitivity:{:.4f}, Specificity: {:.4f}, Dice: {:.4f}, Jaccard: {:.4f}".format(
						operation_point, accuracy, sensitivity, specificity, dice, jaccard))

				# self.train_operating_point.append(operation_point)
				# self.train_aucs.append(roc_auc_train)
				self.epochz.append(epoch)

				self.train_accuracy.append(accuracy)
				self.train_sensitivity.append(sensitivity)
				self.train_specificity.append(specificity)
				self.train_dice.append(dice)
				self.train_jaccard.append(jaccard)

				operation_point, _, _, accuracy, specificity, sensitivity, dice, jaccard = data_utils.use_operating_points(
					operation_point,
					self.y_val.flatten(), self.X_val.flatten())

				print(
					"\nIn: Validation Operating Point:{:.4f}, Accuracy :{:.4f}, Sensitivity:{:.4f}, Specificity: {:.4f}, Dice: {:.4f}, Jaccard: {:.4f}".format(
						operation_point, accuracy, sensitivity, specificity, dice, jaccard))

				self.val_accuracy_in.append(accuracy)
				self.val_sensitivity_in.append(sensitivity)
				self.val_specificity_in.append(specificity)
				self.val_dice_in.append(dice)
				self.val_jaccard_in.append(jaccard)

				operation_point, _, _, accuracy, specificity, sensitivity, dice, jaccard = data_utils.use_operating_points(
					operation_point,
					self.y_val.flatten(), y_pred_val.flatten())

				print(
					"Out: Validation Operating Point:{:.4f}, Accuracy :{:.4f}, Sensitivity:{:.4f}, Specificity: {:.4f}, Dice: {:.4f}, Jaccard: {:.4f}\n".format(
						operation_point, accuracy, sensitivity, specificity, dice, jaccard))

				# Can be used later for drawing plots
				# self.val_operating_point.append(operation_point)
				# self.val_aucs.append(roc_auc_val)
				self.val_accuracy.append(accuracy)
				self.val_sensitivity.append(sensitivity)
				self.val_specificity.append(specificity)
				self.val_dice.append(dice)
				self.val_jaccard.append(jaccard)


				# np.savez(os.path.join(self.logging_dir, self.model_name, 'val_metrics'),
				#          name1=self.val_accuracy,
				#          name2=self.val_sensitivity,
				#          name3=self.val_specificity, name4=self.val_dice)
				#
				# np.savez(os.path.join(self.logging_dir, self.model_name,'train_metrics'),
				#          name1=self.train_accuracy,
				#          name2=self.train_sensitivity,
				#          name3=self.train_specificity, name4=self.train_dice)

				# save as csv file
				df = pandas.DataFrame(data={"epoch": self.epochz, "train_accuracy_in": self.train_accuracy_in,
											"train_sensitivity_in": self.train_sensitivity_in,
											"train_specificity_in": self.train_specificity_in,
											"train_dice_in": self.train_dice_in,"train_jaccard_in": self.train_jaccard_in,
											"val_accuracy_in": self.val_accuracy_in,
											"val_sensitivity_in": self.val_sensitivity_in,
											"val_specificity_in": self.val_specificity_in, "val_dice_in": self.val_dice_in,"val_jaccard_in": self.val_jaccard_in,
											"train_accuracy": self.train_accuracy,
											"train_sensitivity": self.train_sensitivity,
											"train_specificity": self.train_specificity,
											"train_dice": self.train_dice,"train_jaccard": self.train_jaccard,
											"val_accuracy": self.val_accuracy,
											"val_sensitivity": self.val_sensitivity,
											"val_specificity": self.val_specificity, "val_dice": self.val_dice
											})
				df.to_csv(os.path.join(self.logging_dir, 'log2.csv'), sep=',', index=False)

			if self.save_inter_layers_flag == 1:
				# This part is for generating prediction from intermediate epochs
				save_at_epochs = [2,4,6,8,10,12,14,16,18,20]

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





class IntervalEvaluation_cascaded(Callback):
	def __init__(self, model_name, logging_dir, interval, unet_or_srunet,unet_main,validation_data=(), training_data=()):
		super(Callback, self).__init__()
		self.interval = interval
		self.model_name = model_name
		self.logging_dir = logging_dir
		self.unet_or_srunet = unet_or_srunet
		self.unet_main = unet_main

		self.X_val, self.y_val_encoded, self.y_val = validation_data
		self.X_train, self.y_train_encoded, self.y_train = training_data

	def on_train_begin(self, logs={}):
		self.train_accuracy = []
		self.train_sensitivity = []
		self.train_specificity = []
		self.train_dice = []
		self.train_jaccard = []

		self.val_accuracy = []
		self.val_sensitivity = []
		self.val_specificity = []
		self.val_dice = []
		self.val_jaccard = []

		self.epochz = []

	def on_train_end(self, logs={}):
		return

	def on_epoch_begin(self, epoch, logs={}):
		return

	def on_epoch_end(self, epoch, logs={}):

		num = 5

		randomize = np.arange(len(self.X_val))
		np.random.shuffle(randomize)

		y_out_val = self.model.predict([self.X_val[randomize[:num]], self.y_val_encoded[randomize[:num]]], verbose=0)
		y_unet_val = self.unet_main.predict(self.X_val[randomize[:num]], verbose=0)

		randomize_train = np.arange(len(self.X_train))
		np.random.shuffle(randomize_train)

		y_out_train = self.model.predict([self.X_train[randomize_train[:num]], self.y_train_encoded[randomize_train[:num]]], verbose=0)
		y_unet_train = self.unet_main.predict(self.X_train[randomize_train[:num]], verbose=0)


		# SAving predictions per epoch
		figure_path = os.path.join(self.logging_dir, 'figures')
		if not os.path.isdir(figure_path):
			os.mkdir(figure_path)

		# save 5 training and 5 Validation at each epoch
		data_utils.saveResultasPlot_cascade(figure_path, epoch, self.X_train[randomize_train[:num]], self.y_train[randomize_train[:num]],y_unet_train, y_out_train, 'Training', num)
		data_utils.saveResultasPlot_cascade(figure_path, epoch, self.X_val[randomize[:num]], self.y_val[randomize[:num]], y_unet_val, y_out_val, 'Validation', num)


		if epoch % self.interval == 0:
			# For finding AUC
			# fpr, tpr, _ = roc_curve(self.y_train.flatten()>0.5, y_pred_train.flatten())
			# roc_auc_train = auc(fpr, tpr)
			#
			# fpr, tpr, _ = roc_curve(self.y_val.flatten()>0.5, y_pred_val.flatten())
			# roc_auc_val = auc(fpr, tpr)

			y_out_val = self.model.predict([self.X_val, self.y_val_encoded], verbose=0)
			y_out_train = self.model.predict([self.X_train, self.y_train_encoded], verbose=0)


			operation_point, _, _, accuracy, specificity, sensitivity, dice, jaccard = data_utils.get_operating_points(
				self.y_train.flatten(), y_out_train.flatten())

			print(
				"\nOut: Training Operating Point:{:.4f}, Accuracy :{:.4f}, Sensitivity:{:.4f}, Specificity: {:.4f}, Dice: {:.4f}, Jaccard: {:.4f}".format(
					operation_point, accuracy, sensitivity, specificity, dice, jaccard))

			# self.train_operating_point.append(operation_point)
			# self.train_aucs.append(roc_auc_train)
			self.epochz.append(epoch)

			self.train_accuracy.append(accuracy)
			self.train_sensitivity.append(sensitivity)
			self.train_specificity.append(specificity)
			self.train_dice.append(dice)
			self.train_jaccard.append(jaccard)


			operation_point, _, _, accuracy, specificity, sensitivity, dice, jaccard = data_utils.use_operating_points(
				operation_point,
				self.y_val.flatten(), y_out_val.flatten())

			print(
				"Out: Validation Operating Point:{:.4f}, Accuracy :{:.4f}, Sensitivity:{:.4f}, Specificity: {:.4f}, Dice: {:.4f}, Jaccard: {:.4f}\n".format(
					operation_point, accuracy, sensitivity, specificity, dice, jaccard))

			# Can be used later for drawing plots
			# self.val_operating_point.append(operation_point)
			# self.val_aucs.append(roc_auc_val)
			self.val_accuracy.append(accuracy)
			self.val_sensitivity.append(sensitivity)
			self.val_specificity.append(specificity)
			self.val_dice.append(dice)
			self.val_jaccard.append(jaccard)

			# np.savez(os.path.join(self.logging_dir, self.model_name, 'val_metrics'),
			#          name1=self.val_accuracy,
			#          name2=self.val_sensitivity,
			#          name3=self.val_specificity, name4=self.val_dice)
			#
			# np.savez(os.path.join(self.logging_dir, self.model_name,'train_metrics'),
			#          name1=self.train_accuracy,
			#          name2=self.train_sensitivity,
			#          name3=self.train_specificity, name4=self.train_dice)

			# save as csv file
			df = pandas.DataFrame(data={"epoch": self.epochz, "train_accuracy": self.train_accuracy,
										"train_sensitivity": self.train_sensitivity,
										"train_specificity": self.train_specificity, "train_dice": self.train_dice, "train_jaccard": self.train_jaccard,
										"val_accuracy": self.val_accuracy, "val_sensitivity": self.val_sensitivity,
										"val_specificity": self.val_specificity, "val_dice": self.val_dice, "val_jaccard": self.val_jaccard})
			df.to_csv(os.path.join(self.logging_dir, 'log2.csv'), sep=',', index=False)




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
    
    
class DataGenerator_Augment(keras.utils.Sequence):
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

        seq = iaa.Sequential([
            iaa.Fliplr(0.5),  # Flip Y-axis
            iaa.TranslateX(px=(-20, 20)),  # Translate along X axis by 20-20 pixels
            iaa.TranslateY(px=(-20, 20)),  # Trasnlate Y
            iaa.Rotate((-20, 20))  # Rotate
            # iaa.ScaleX((0.5, 1.5)), # Along width 50%-150% of size
            # iaa.ScaleY((0.5, 1.5)), # Along height
            # iaa.Pepper(0.1), # Replace 10% of pixel with blackish colors
            # iaa.Salt(0.1), # Whiteish colors
            # iaa.GaussianBlur(sigma=(0, 3.0))
            ], random_order=True)

        counter = 0
        RESIZE_DIM = X.shape[1]
        RESIZE_DIM_ = X.shape[2]
        channels = X.shape[-1]
        #print(RESIZE_DIM, RESIZE_DIM_, channels)
        X_values_augmented = []
        Y_values_augmented = []
        
        #print(np.unique(y[0]))
        for a,b in zip(X, y):
            for p in range(1):
                
                #print(a.shape, b.shape)
                #images_aug = seq.augment_images(a.reshape(1,RESIZE_DIM,RESIZE_DIM_,channels))
                #masks_aug = seq.augment_images(b.reshape(1,RESIZE_DIM,RESIZE_DIM_,1))
                
                images_aug, masks_aug = seq(images=a.reshape(1,RESIZE_DIM,RESIZE_DIM_,channels), segmentation_maps=b.reshape(1,RESIZE_DIM,RESIZE_DIM_,1).astype('int16'))
                
                #print(images_aug.shape, masks_aug.shape)
                
                X_values_augmented.append( images_aug.reshape(RESIZE_DIM,RESIZE_DIM_,channels))
                Y_values_augmented.append( masks_aug.reshape(RESIZE_DIM,RESIZE_DIM_,1))

            counter = counter + 1


        # prev number of images = n
        # augmented number of images = n * 4 ( 2 seq 2 times)
        X_values_augmented = np.asarray( X_values_augmented )
        Y_values_augmented = np.asarray( Y_values_augmented )
        
        X = np.concatenate( (X, X_values_augmented), axis = 0)
        y = np.concatenate( (y, Y_values_augmented), axis = 0)
        
        # Normalize data to [0-1]
        # X = X.astype('float32')
        # X /= 255
    
        return X, y


class DataGenerator_Augment_cascaded(keras.utils.Sequence):
	'Generates data for Keras'

	def __init__(self, Xdata,srunet, Ydata, batch_size=32, shuffle=True):
		'Initialization'
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.Xdata = Xdata
		self.srunet = srunet
		self.Ydata = Ydata
		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.Xdata) / self.batch_size))

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
		# Generate data
		X, y = self.__data_generation(indexes)

		return X, y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.Xdata))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def __data_generation(self, indexes):
		'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
		# Generate data
		X = self.Xdata[indexes]
		y = self.Ydata[indexes]

		seq = iaa.Sequential([
			iaa.Fliplr(0.5),  # Flip Y-axis
			iaa.TranslateX(px=(-20, 20)),  # Translate along X axis by 20-20 pixels
			iaa.TranslateY(px=(-20, 20)),  # Trasnlate Y
			iaa.Rotate((-20, 20)),  # Rotate
			iaa.ScaleX((0.5, 1.5)), # Along width 50%-150% of size
			iaa.ScaleY((0.5, 1.5)), # Along height
			# iaa.Pepper(0.1), # Replace 10% of pixel with blackish colors
			# iaa.Salt(0.1), # Whiteish colors
			iaa.GaussianBlur(sigma=(0, 3.0))
		], random_order=True)

		counter = 0
		RESIZE_DIM = X.shape[1]
		RESIZE_DIM_ = X.shape[2]
		channels = X.shape[-1]
		# print(RESIZE_DIM, RESIZE_DIM_, channels)
		X_values_augmented = []
		Y_values_augmented = []

		# print(np.unique(y[0]))
		for a, b in zip(X, y):
			for p in range(1):
				# print(a.shape, b.shape)
				# images_aug = seq.augment_images(a.reshape(1,RESIZE_DIM,RESIZE_DIM_,channels))
				# masks_aug = seq.augment_images(b.reshape(1,RESIZE_DIM,RESIZE_DIM_,1))

				images_aug, masks_aug = seq(images=a.reshape(1, RESIZE_DIM, RESIZE_DIM_, channels),
											segmentation_maps=b.reshape(1, RESIZE_DIM, RESIZE_DIM_, 1).astype('int16'))

				# print(images_aug.shape, masks_aug.shape)

				X_values_augmented.append(images_aug.reshape(RESIZE_DIM, RESIZE_DIM_, channels))
				Y_values_augmented.append(masks_aug.reshape(RESIZE_DIM, RESIZE_DIM_, 1))

			counter = counter + 1

		# prev number of images = n
		# augmented number of images = n * 4 ( 2 seq 2 times)
		X_values_augmented = np.asarray(X_values_augmented)
		Y_values_augmented = np.asarray(Y_values_augmented)

		#X = np.concatenate((X, X_values_augmented), axis=0)
		#y = np.concatenate((y, Y_values_augmented), axis=0)
		X = X_values_augmented
		y = Y_values_augmented
		#print(np.shape(y))
		y_encoded, _ = self.srunet.predict(x=y, batch_size=16, verbose=2)

		#print(np.shape(y_encoded))

		# Normalize data to [0-1]
		# X = X.astype('float32')
		# X /= 255
		X = [X, y_encoded]

		y = y#comment this out for normal cascade. This was to monitor the UNET results as well as shape regularization results

		return X, y