# Tajwar Abrar Aleef 07/2020
# Data analysis and preprocessing for TN-SCUI2020

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

def plot_results(X, Y, path, suffix):

    plt.figure()
    ax = plt.subplot(1, 2, 1)
    plt.imshow(X, cmap='gray')
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel('Input: {}'.format(np.shape(X)))
    ax.get_xaxis().set_visible(True)
    ax.get_xaxis().set_ticks([])
    ax.get_xaxis().set_ticklabels([])

    ax = plt.subplot(1, 2, 2)
    plt.imshow(Y, cmap='gray')
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel('Pred: {}'.format(np.shape(Y)))
    ax.get_xaxis().set_visible(True)
    ax.get_xaxis().set_ticks([])
    ax.get_xaxis().set_ticklabels([])

    plt.savefig(os.path.join(path, "%s.png" % (suffix)))
    plt.clf()
    plt.close()


path = '../datasets/test/'
model_name = 'data0_Cascaded_efb3_loss_augmentation_8_adversarial'
x_dim, y_dim = 256, 256
number_of_samples = 910
sanity_check = 0

Xdata = np.zeros((number_of_samples, x_dim, y_dim))
dimensions = np.zeros((number_of_samples, 2))

# Loading data and keeping track of dimension
print("Loading results....")
for pid in range(number_of_samples):
    temp = (cv2.imread(path + 'test_' + str(pid + 1) +'.PNG',0)) #file starts from 1
    dimensions[pid, :] = np.shape(temp)
    Xdata[pid] = cv2.resize(temp, (x_dim, y_dim), interpolation=cv2.INTER_LINEAR)

#Preprocessing step
Xdata = np.expand_dims(Xdata, axis=-1)/255

print("X Test set size -------> ", Xdata.shape)
print("X Test set -> max: %s, min: %s" %(np.max(Xdata), np.min(Xdata)))

#Load model
weights_path = "../logs/{}/{}.h5".format(model_name, model_name)
model = None
model = load_model(weights_path, compile=False)
Ydata = model.predict(x=Xdata, batch_size=16, verbose=1)


#Resizing
output_path = '../logs/{}/test_output/'.format(model_name)
create_directory(output_path)

sanity_check_path = '../logs/{}/test_output_sanity/'.format(model_name)
if sanity_check == 1:
    create_directory(sanity_check_path)

for pid in range(number_of_samples):

    print(pid)
    temp = Ydata[pid,:,:,:] #file starts from 1
    temp_dimension = dimensions[pid]
    temp = cv2.resize(temp, (int(temp_dimension[1]), int(temp_dimension[0])), interpolation=cv2.INTER_LINEAR)
    temp = np.float32(temp>=0.5)*255

    #print(temp.shape)
    #print("Y -> max: %s, min: %s" % (np.max(temp), np.min(temp)))


    cv2.imwrite("{}/{}.PNG".format(output_path, 'test_' + str(pid + 1)), temp)

    if sanity_check == 1:
            Xoriginal = (cv2.imread(path + 'test_' + str(pid + 1) + '.PNG', 0))
            plot_results(Xoriginal, temp, sanity_check_path, (pid+1))



print('debug')






