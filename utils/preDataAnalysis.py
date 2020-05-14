import os
import csv
import numpy as np
import cv2
from scipy import stats

path = 'D:/Projects/Challenges/TN-SCUI2020/Data/train/'

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
Xdata = []
dimension = np.zeros((len(fileNames), 2))

for pid in range(len(fileNames)):
	Xdata.append(cv2.imread(path + 'image/' + fileNames[pid],0))
	dimension[pid, :] = np.shape(Xdata[pid])


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
#most of the data and introducing not existing pixel values.

#This however can be later changed if required. But for now resizing all images to around 256x256 seems reasonable

