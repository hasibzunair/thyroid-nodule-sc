# Tajwar Abrar Aleef 05/2020
# Data analysis and preprocessing for TN-SCUI2020

import os
import csv
import numpy as np
import cv2
from scipy import stats
import matplotlib.pyplot as plt

path = 'D:/Projects/Challenges/TN-SCUI2020/Data/train/'

x_dim = y_dim = 256



with open(path + 'train.csv', 'r') as file:
	reader = csv.reader(file)
	fileNames = []
	for row in reader:
		fileNames.append(row)

# removing header info
fileNames.pop(0)

# converting to array
fileNames = np.asarray(fileNames)
Ydata = np.uint8(fileNames[:, 1])  # this is the GT for classification
fileNames = fileNames[:, 0]

print('Total number of samples: ' + str(len(fileNames)))
print('Total number of  positive samples: ' + str(np.sum(Ydata)))
print('Total number of  negative samples: ' + str(np.sum(np.logical_not(Ydata))))

# dimensionality analysis
Xdata = np.zeros((len(fileNames), x_dim, y_dim))
Xmask = np.zeros((len(fileNames), x_dim, y_dim))

dimension = np.zeros((len(fileNames), 2))

for pid in range(len(fileNames)):
	temp = (cv2.imread(path + 'image/' + fileNames[pid],0))
	mask = (cv2.imread(path + 'mask/' + fileNames[pid], 0))

	dimension[pid, :] = np.shape(temp)

	Xdata[pid] = cv2.resize(temp, (x_dim, y_dim), interpolation=cv2.INTER_LINEAR)
	Xmask[pid] = cv2.resize(mask, (x_dim, y_dim), interpolation=cv2.INTER_LINEAR)


#finding the most common unique dimension
unique_dim, indx = np.unique(dimension, return_counts=1, axis=0)

print("Total number of different resolutions: " + str(len(unique_dim)))
print("Most common resolution: " + str(unique_dim[indx == max(indx)]) + ', Occuring for ' + str(max(indx)) + ' times')

#finding the max resolution and min resolution
print("\nMax Resolutions:")
print(str(unique_dim[unique_dim[:, 0] == np.max(unique_dim, axis=0)[0]]))
print(str(unique_dim[unique_dim[:, 1] == np.max(unique_dim, axis=0)[1]]))

print("\nMin Resolutions:")
print(str(unique_dim[unique_dim[:, 0] == np.min(unique_dim, axis=0)[0]]))
print(str(unique_dim[unique_dim[:, 1] == np.min(unique_dim, axis=0)[1]]))

print("\nMean Resolution:")
print(np.mean(dimension, axis=0))

#Since the minimum resolution observed by itself is quite large 206 x 313 and 319 x 247, we should
#pick the resizing dimension somewhere between this line as it will ensure we are not upsampling
#most of the data and introducing non-existing pixel values.

#This however can be later changed if required. But for now resizing all images to around 256x256 seems reasonable


# Saving Data
Xdata_org = Xdata
#Only resized data
np.savez(path + '/npy_data/data0.npz', name1=Xdata, name2=Xmask, name3=Ydata)

#Histogram normalized data
for i in range(len(Xdata)):
	Xdata[i] = cv2.equalizeHist(Xdata_org[i].astype(np.uint8))

np.savez(path + '/npy_data/data1.npz', name1=Xdata, name2=Xmask, name3=Ydata)

#Contrast Limited Adaptive Histogram Equalization (CLAHE)
for i in range(len(Xdata)):

	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
	Xdata[i] = clahe.apply(Xdata_org[i].astype(np.uint8))

np.savez(path + '/npy_data/data2.npz', name1=Xdata, name2=Xmask, name3=Ydata)


# Checking pre-processed data
data = np.load(path + '/npy_data/data0.npz')
Xdata = data['name1']
Xmask = data['name2']
Ydata = data['name3']

data = np.load(path + '/npy_data/data1.npz')
Xdata_hist_norm = data['name1']

data = np.load(path + '/npy_data/data2.npz')
Xdata_clahe = data['name1']


# visualizing preprocessed data
rand_number = 20 #just a random starting indx to plot the next 5 image data
r = 5
c = 3
plt.figure(figsize=(c + 20, r + 10))
for i in range(r):
	ax = plt.subplot(r, c, 1 + c*(i))
	plt.imshow(Xdata[i + rand_number], cmap='gray')
	ax.set_xticks([])
	ax.set_yticks([])
	if(i == 0): plt.title('Org')

	ax = plt.subplot(r, c, 2 + c*(i))
	plt.imshow(Xdata_hist_norm[i + rand_number], cmap='gray')
	ax.set_xticks([])
	ax.set_yticks([])
	if(i == 0): plt.title('Hist_norm')

	ax = plt.subplot(r, c, 3 + c*(i))
	plt.imshow(Xdata_clahe[i + rand_number], cmap='gray')
	ax.set_xticks([])
	ax.set_yticks([])
	if(i == 0): plt.title('Clahe')

