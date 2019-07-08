
import numpy as np
from scipy import optimize
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.svm import SVC
import csv
import re
from itertools import cycle
from random import choices
from itertools import islice
from scipy import interp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, auc





def getTrainTest():
    input=[]
    with open('messidor.csv', 'r') as f:
        for line in islice(f, 1, 1152):
            currentline = line.split(",")
            currentline = list(map(float, currentline))
            print(currentline)
            input.append(currentline)


    print(len(input))
    # input_list = choices(input, k=int(len(input) * 0.8))
    input = np.array(input)
    np.random.shuffle(input)
    print(len(input))

    input_training = input[:int(0.6*len(input))]
    input_testing = input[int(0.6*len(input)):]
    with open('training.data', 'w') as f:
        for line in input_training:
            for i in range(0,19):
                f.write(str(line[i])+",")
            f.write(str(line[19]))
            f.write("\n")

    with open('testing.data', 'w') as f:
        for line in input_testing:
            for i in range(0,19):
                f.write(str(line[i])+",")
            f.write(str(line[19]))
            f.write("\n")
    #shuffle data

def svm1():
    input_list = []
    with open('training.data', 'r') as f:
        for line in f:
            currentline = line.split(",")
            currentline = list(map(float, currentline))
            input_list.append(currentline)

    training = np.array(input_list)
    training_x = training[:, :-1]
    training_y = training[:, -1:]

    input_list_testing = []
    with open('testing.data', 'r') as f:
        for line in f:
            currentline = line.split(",")
            currentline = list(map(float, currentline))
            input_list_testing.append(currentline)

    testing = np.array(input_list_testing)
    testing_x = testing[:, :-1]
    print(testing_x)
    testing_y = testing[:, -1:]
    testing_y_trans = np.ravel(testing_y)
    training_y_trans = np.ravel(training_y)


    sv = SVC(C=1, kernel='linear')

    # train the model
    sv.fit(training_x, training_y_trans)

    # predict the labels and report accuracy
    hard_pred = sv.predict(testing_x)


    acc = np.isclose(hard_pred, testing_y_trans).sum() / len(hard_pred)
    print("Testing accuracy: {}".format(acc))
    print(hard_pred)

    hard_pred_training = sv.predict(training_x)
    acc = np.isclose(hard_pred_training, training_y_trans).sum() / len(hard_pred_training)
    print("Training accuracy: {}".format(acc))

    print(sv.coef_)



def svm2():
    input_list = []
    with open('training.data', 'r') as f:
        for line in f:
            currentline = line.split(",")
            currentline = list(map(float, currentline))
            input_list.append(currentline)

    training = np.array(input_list)
    training_x = training[:, :-1]
    training_y = training[:, -1:]

    input_list_testing = []
    with open('testing.data', 'r') as f:
        for line in f:
            currentline = line.split(",")
            currentline = list(map(float, currentline))
            input_list_testing.append(currentline)

    testing = np.array(input_list_testing)
    testing_x = testing[:, :-1]
    testing_y = testing[:, -1:]

    trainning_err = []
    testing_acc = []
    x_axis = [10e-12, 10e-6, 1, 10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    testing_y_trans = np.ravel(testing_y)
    training_y_trans = np.ravel(training_y)


    for c in x_axis:

        sv = SVC(C = c, kernel='linear')



        # train the model
        sv.fit(training_x, training_y_trans)

        # predict the labels and report accuracy
        hard_pred = sv.predict(testing_x)


        acc = np.isclose(hard_pred, testing_y_trans).sum() / len(hard_pred)
        testing_acc.append(acc)
        print("Testing accuracy: {}".format(acc))

        hard_pred_training = sv.predict(training_x)
        acc = np.isclose(hard_pred_training, training_y_trans).sum() / len(hard_pred_training)
        print("Training accuracy: {}".format(acc))

        trainning_err_cur = 0;

        for i in range(0, len(hard_pred_training)):
            trainning_err_cur += max(0, 1-training_y_trans[i]*hard_pred_training[i])
        trainning_err.append(trainning_err_cur)


    plt.plot(x_axis, trainning_err)

    plt.show()

    plt.plot(x_axis, testing_acc)

    plt.show()

    # use predicted probabilities to construct ROC curve and AUC score
    # soft_pred = sv.predict_proba(test.iloc[:, 1:])
    # fpr, tpr, thresh = roc_curve(test.iloc[:, 0], soft_pred[:, 1])
    # auc = roc_auc_score(test.iloc[:, 0], soft_pred[:, 1])
    # print("ROC Curve:")
    # plt.plot(fpr, tpr)
    # plt.plot([0, 1], [0, 1], "r--", alpha=.5)
    # plt.show()
    # print("AUC: {}".format(auc))

if __name__ == '__main__':
    #getTrainTest()
    svm1()
    #svm2()





    # SUPPORT VECTOR MACHINE
