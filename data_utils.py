import numpy as np
import keras
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
import h5py
import pickle
import os
import io
import math
from sklearn.metrics import roc_curve, auc



# import matplotlib as mpl
# mpl.use('Agg')

def saveCorruptionResults(figure_path, epoch, X, Y, filename, num = 5):


    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    X = X[randomize] #pred
    Y = Y[randomize] #train
    #Plotting the first *num* cases

    X = X[:num]
    Y = Y[:num]


    for j in range(num): #how many validation images to save (take care of the batch size, if batch size is less than this number then it wont work)

        X_temp = X[j]
        Y_temp = Y[j]

        operation_point, _, _, accuracy, specificity, sensitivity, dice = get_operating_points(Y_temp.flatten(),
                                                                                               X_temp.flatten())

        rows = 1
        cols = 4  # how many examples to be plotted
        plt.figure(figsize=(cols*3, rows*3))

        cnt = 1
        # display input
        ax = plt.subplot(rows, cols, cnt)
        plt.imshow(X_temp[:,:,0], cmap = 'gray')
        ax.get_yaxis().set_visible(False)
        ax.set_xlabel('Input')
        ax.get_xaxis().set_visible(True)
        ax.get_xaxis().set_ticks([])
        ax.get_xaxis().set_ticklabels([])

        cnt += 1
        ax = plt.subplot(rows, cols, cnt)
        plt.imshow(Y_temp[:,:,0],cmap = 'gray')
        ax.get_yaxis().set_visible(False)
        ax.set_xlabel('GT')
        ax.get_xaxis().set_visible(True)
        ax.get_xaxis().set_ticks([])
        ax.get_xaxis().set_ticklabels([])


        cnt += 1
        ax = plt.subplot(rows, cols, cnt)
        plt.imshow(X_temp[:,:,0]>0.5,cmap = 'gray')
        ax.get_yaxis().set_visible(False)
        ax.set_xlabel('Input Thrs')
        ax.get_xaxis().set_visible(True)
        ax.get_xaxis().set_ticks([])
        ax.get_xaxis().set_ticklabels([])

        cnt += 1
        ax = plt.subplot(rows, cols, cnt)
        plt.imshow(Y_temp[:,:,0], 'gray', interpolation='none')
        plt.imshow(X_temp[:,:,0]>0.5, 'jet', interpolation='none', alpha=0.7)
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(True)
        ax.get_xaxis().set_ticks([])
        ax.get_xaxis().set_ticklabels([])
        ax.set_xlabel('Diff')


        plt.suptitle(
            'Accuracy:{:.4f}, Sensitivity:{:.4f}, Specificity:{:.4f}, Dice:{:.4f}, Operating Point:{:.4f}'.format(
                accuracy, sensitivity, specificity, dice, operation_point))
        plt.savefig(os.path.join(figure_path, "%s_epoch_%s_no_%s.png" % (filename, epoch, str(j))))
        plt.clf()
        plt.close()


    return


def saveResultasPlot(figure_path, epoch, X, Y, Y_pred, filename, num = 5):


    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    X = X[randomize[:num]]
    Y = Y[randomize[:num]]
    Y_pred = Y_pred[randomize[:num]]
    #Plotting the first *num* cases

    # X = X[:num]
    # Y = Y[:num]
    # Y_pred = Y_pred[:num]




    for j in range(num): #how many validation images to save (take care of the batch size, if batch size is less than this number then it wont work)

        X_temp = X[j]
        Y_temp = Y[j]
        Y_pred_temp = Y_pred[j]

        operation_point, _, _, accuracy, specificity, sensitivity, dice = get_operating_points(Y_temp.flatten(),
                                                                                               Y_pred_temp.flatten())

        rows = 1
        cols = 5  # how many examples to be plotted
        plt.figure(figsize=(cols*3, rows*3))

        cnt = 1
        # display input
        ax = plt.subplot(rows, cols, cnt)
        plt.imshow(X_temp[:,:,0],cmap = 'gray')
        ax.get_yaxis().set_visible(False)
        ax.set_xlabel('Input')
        ax.get_xaxis().set_visible(True)
        ax.get_xaxis().set_ticks([])
        ax.get_xaxis().set_ticklabels([])

        cnt += 1
        ax = plt.subplot(rows, cols, cnt)
        plt.imshow(Y_temp[:,:,0],cmap = 'gray')
        ax.get_yaxis().set_visible(False)
        ax.set_xlabel('GT')
        ax.get_xaxis().set_visible(True)
        ax.get_xaxis().set_ticks([])
        ax.get_xaxis().set_ticklabels([])

        cnt += 1
        ax = plt.subplot(rows, cols, cnt)
        plt.imshow(Y_pred_temp[:,:,0],cmap = 'gray')
        ax.get_yaxis().set_visible(False)
        ax.set_xlabel('Pred')
        ax.get_xaxis().set_visible(True)
        ax.get_xaxis().set_ticks([])
        ax.get_xaxis().set_ticklabels([])

        cnt += 1
        ax = plt.subplot(rows, cols, cnt)
        plt.imshow(Y_pred_temp[:,:,0]>0.5,cmap = 'gray')
        ax.get_yaxis().set_visible(False)
        ax.set_xlabel('Thr. Pred')
        ax.get_xaxis().set_visible(True)
        ax.get_xaxis().set_ticks([])
        ax.get_xaxis().set_ticklabels([])

        cnt += 1
        ax = plt.subplot(rows, cols, cnt)
        plt.imshow(Y_temp[:,:,0], 'gray', interpolation='none')
        plt.imshow(Y_pred_temp[:,:,0]>0.5, 'jet', interpolation='none', alpha=0.7)
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(True)
        ax.get_xaxis().set_ticks([])
        ax.get_xaxis().set_ticklabels([])
        ax.set_xlabel('Diff')


        plt.suptitle(
            'Accuracy:{:.4f}, Sensitivity:{:.4f}, Specificity:{:.4f}, Dice:{:.4f}, Operating Point:{:.4f}'.format(
                accuracy, sensitivity, specificity, dice, operation_point))
        plt.savefig(os.path.join(figure_path, "%s_epoch_%s_no_%s.png" % (filename, epoch, str(j))))
        plt.clf()
        plt.close()


    return

def perf_measure(y_actual, y_hat):

    TP = np.logical_and(y_actual,y_hat)
    FP = np.logical_and(y_hat,abs(y_actual-1))
    TN = np.logical_and(abs(y_hat-1),abs(y_actual-1))
    FN = np.logical_and(y_actual,abs(y_hat-1))

    return(np.sum(TP), np.sum(FP), np.sum(TN), np.sum(FN))

def get_operating_points(y_gt, y_pr, threshold_step = 0.001):

    # fpr = []
    # tpr = []
    # threshold = 0
    # distance = 100
    #
    # for k in range(0,int(1/threshold_step)):
    #     y_temp = y_pr>=threshold
    #     TP,FP,TN,FN = perf_measure(y_gt,y_temp)
    #     fpr_temp = (FP / (FP + TN))
    #     tpr_temp = (TP / (TP + FN))
    #
    #     current_distance = math.sqrt((0-fpr_temp)**2 + (1-tpr_temp)**2)
    #
    #     if(current_distance<distance):
    #         distance = current_distance
    #         operation_point_temp = threshold
    #
    #     fpr.append(fpr_temp)
    #     tpr.append(tpr_temp)
    #     threshold = threshold+threshold_step
    #
    # operation_point = (operation_point_temp)
    # roc_auc = auc(fpr, tpr)
    # avg_auc = (roc_auc)

    #manually setting OP as 0.5
    operation_point = 0.5
   #Find Accuracy, Specificity and Sensitivity
    y_temp = y_pr >= operation_point
    TP, FP, TN, FN = perf_measure(y_gt, y_temp)
    accuracy = (TP + TN)/(TP+FP+TN+FN)
    fpr_tempp = (FP / (FP + TN))
    tpr_tempp = (TP / (TP + FN))

    specificity = ((1-fpr_tempp))
    sensitivity = (tpr_tempp)
    dice = (2*TP)/((2*TP)+FP+FN)

    return operation_point, fpr_tempp, tpr_tempp, accuracy,specificity, sensitivity, dice


def use_operating_points(operation_point, y_gt, y_pr, threshold_step = 0.001):
    # fpr = []
    # tpr = []
    # threshold = 0
    #
    # for k in range(0,int(1/threshold_step)):
    #
    #     y_temp = y_pr>=threshold
    #     TP,FP,TN,FN = perf_measure(y_gt,y_temp)
    #     fpr_temp = (FP / (FP + TN))
    #     tpr_temp = (TP / (TP + FN))
    #
    #     fpr.append(fpr_temp)
    #     tpr.append(tpr_temp)
    #     threshold = threshold+threshold_step
    #
    #
    # roc_auc = auc(fpr, tpr)
    # avg_auc = (roc_auc)

    #Find Accuracy, Specificity and Sensitivity
    y_temp = y_pr >= operation_point
    TP, FP, TN, FN = perf_measure(y_gt, y_temp)
    accuracy = (TP + TN)/(TP+FP+TN+FN)
    fpr_tempp = (FP / (FP + TN))
    tpr_tempp = (TP / (TP + FN))

    specificity = ((1-fpr_tempp))
    sensitivity = (tpr_tempp)
    dice = (2*TP)/((2*TP)+FP+FN)

    return operation_point, fpr_tempp, tpr_tempp, accuracy,specificity, sensitivity, dice