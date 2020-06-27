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



def saveResultasPlot_cascade(figure_path, epoch, X, Y, Y_pred1,Y_pred2, filename, num = 5):


    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    X = X[randomize[:num]]
    Y = Y[randomize[:num]]
    Y_pred1= Y_pred1[randomize[:num]]
    Y_pred2 = Y_pred2[randomize[:num]]


    for j in range(num): #how many validation images to save (take care of the batch size, if batch size is less than this number then it wont work)

        X_temp = X[j]
        Y_temp = Y[j]
        Y_pred_temp1 = Y_pred1[j]
        Y_pred_temp2 = Y_pred2[j]

        rows = 1
        cols = 6  # how many examples to be plotted
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
        plt.imshow(Y_pred_temp1[:,:,0],cmap = 'gray')
        ax.get_yaxis().set_visible(False)
        ax.set_xlabel('Unet')
        ax.get_xaxis().set_visible(True)
        ax.get_xaxis().set_ticks([])
        ax.get_xaxis().set_ticklabels([])


        cnt += 1
        ax = plt.subplot(rows, cols, cnt)
        plt.imshow(Y_pred_temp2[:,:,0],cmap = 'gray')
        ax.get_yaxis().set_visible(False)
        ax.set_xlabel('Out')
        ax.get_xaxis().set_visible(True)
        ax.get_xaxis().set_ticks([])
        ax.get_xaxis().set_ticklabels([])

        cnt += 1
        ax = plt.subplot(rows, cols, cnt)
        plt.imshow(Y_pred_temp1[:,:,0]>0.5, 'gray', interpolation='none')
        plt.imshow(Y_pred_temp2[:,:,0]>0.5, 'jet', interpolation='none', alpha=0.7)
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(True)
        ax.get_xaxis().set_ticks([])
        ax.get_xaxis().set_ticklabels([])
        ax.set_xlabel('UnetvsOut')


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
        plt.imshow(Y_temp[:,:,0], 'gray', interpolation='none')
        plt.imshow(Y_pred_temp2[:,:,0]>0.5, 'jet', interpolation='none', alpha=0.7)
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(True)
        ax.get_xaxis().set_ticks([])
        ax.get_xaxis().set_ticklabels([])
        ax.set_xlabel('OutVsGt')

        operation_point1, _, _, accuracy1, specificity1, sensitivity1, dice1 = get_operating_points(Y_temp.flatten(),
                                                                                               Y_pred_temp1.flatten())

        operation_point2, _, _, accuracy2, specificity2, sensitivity2, dice2 = get_operating_points(Y_temp.flatten(),
                                                                                               Y_pred_temp2.flatten())

        plt.suptitle(
            'Unet: Accuracy:{:.4f}, Sensitivity:{:.4f}, Specificity:{:.4f}, Dice:{:.4f}, Operating Point:{:.4f} '
            '\nOut: Accuracy:{:.4f}, Sensitivity:{:.4f}, Specificity:{:.4f}, Dice:{:.4f}, Operating Point:{:.4f}'.format(
                accuracy1, sensitivity1, specificity1, dice1, operation_point1, accuracy2, sensitivity2, specificity2, dice2, operation_point2))
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


# %%
# Plot and save accuravy loss graphs individually
def plot_loss_accu(history,LOG_PATH, EXPERIMENT_NAME):
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
    # plt.show()

    loss = history.history['jacard'][1:]
    val_loss = history.history['val_jacard'][1:]
    epochs = range(len(loss))
    plt.plot(epochs, loss, 'r')
    plt.plot(epochs, val_loss, 'b')
    plt.title('Training and validation jaccard index')
    plt.ylabel('Jaccard Index %')
    # plt.xlabel('Epoch')
    plt.legend(['training', 'validation'], loc='lower right')
    plt.grid(True)
    plt.savefig('{}/{}_jac.png'.format(LOG_PATH, EXPERIMENT_NAME), dpi=100)
    # plt.show()

    loss = history.history['dice'][1:]
    val_loss = history.history['val_dice'][1:]
    epochs = range(len(loss))
    plt.plot(epochs, loss, 'r')
    plt.plot(epochs, val_loss, 'b')
    plt.title('Training and validation dice')
    plt.ylabel('Dice Score %')
    # plt.xlabel('Epoch')
    plt.legend(['training', 'validation'], loc='lower right')
    plt.grid(True)
    plt.savefig('{}/{}_jac.png'.format(LOG_PATH, EXPERIMENT_NAME), dpi=100)
    # plt.show()


def plot_graphs(history,LOG_PATH, EXPERIMENT_NAME):
    plt.figure(figsize=(18, 5))
    plt.subplot(131)
    plt.plot(history.history['loss'], linewidth=4)
    plt.plot(history.history['val_loss'], linewidth=4)
    plt.title('Loss')
    # plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid(True)

    plt.subplot(132)
    plt.plot(history.history['jacard'], linewidth=4)
    plt.plot(history.history['val_jacard'], linewidth=4)
    plt.title('Jacard')
    # plt.ylabel('Jacard')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.subplot(133)
    plt.plot(history.history['dice_coef'], linewidth=4)
    plt.plot(history.history['val_dice_coef'], linewidth=4)
    plt.title('Dice')
    # plt.ylabel('Dice')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid(True)
    plt.savefig('{}/{}_graph.png'.format(LOG_PATH, EXPERIMENT_NAME), dpi=100)
    # plt.show()


def plot_graphs_cascade(history,LOG_PATH, EXPERIMENT_NAME):
    plt.figure(figsize=(18, 18))
    plt.subplot(231)
    plt.plot(history.history['model_1_loss'], linewidth=4)
    plt.plot(history.history['val_model_1_loss'], linewidth=4)
    plt.title('Loss Unet')
    # plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid(True)

    plt.subplot(232)
    plt.plot(history.history['model_1_jacard'], linewidth=4)
    plt.plot(history.history['val_model_1_jacard'], linewidth=4)
    plt.title('Jacard Unet')
    # plt.ylabel('Jacard')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.subplot(233)
    plt.plot(history.history['model_1_dice_coef'], linewidth=4)
    plt.plot(history.history['val_model_1_dice_coef'], linewidth=4)
    plt.title('Dice Unet')
    # plt.ylabel('Dice')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid(True)
    plt.savefig('{}/{}_graph.png'.format(LOG_PATH, EXPERIMENT_NAME), dpi=100)

    plt.subplot(234)
    plt.plot(history.history['model_2_loss'], linewidth=4)
    plt.plot(history.history['val_model_2_loss'], linewidth=4)
    plt.title('Loss Out')
    # plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid(True)

    plt.subplot(235)
    plt.plot(history.history['model_2_jacard'], linewidth=4)
    plt.plot(history.history['val_model_2_jacard'], linewidth=4)
    plt.title('Jacard Out')
    # plt.ylabel('Jacard')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.subplot(236)
    plt.plot(history.history['model_2_dice_coef'], linewidth=4)
    plt.plot(history.history['val_model_2_dice_coef'], linewidth=4)
    plt.title('Dice Out')
    # plt.ylabel('Dice')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid(True)
    plt.savefig('{}/{}_graph.png'.format(LOG_PATH, EXPERIMENT_NAME), dpi=100)
    # plt.show()