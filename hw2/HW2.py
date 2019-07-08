import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

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

#age, size
def getTrainTest():
    input=[]
    with open('ProPublica_COMPAS_preprocessed.csv', 'r') as f:
        for line in islice(f, 1, 5728):
            currentline = line.split(",")
            currentline = list(map(int, currentline))
            print(currentline)
            input.append(currentline)


    print(len(input))
    # input_list = choices(input, k=int(len(input) * 0.8))
    input = np.array(input)
    np.random.shuffle(input)
    print(len(input))

    input_training = input[:int(0.8*len(input))]
    input_testing = input[int(0.8*len(input)):]
    with open('training.data', 'w') as f:
        for line in input_training:
            for i in range(1,12):
                f.write(str(line[i])+",")
            f.write(str(line[12]))
            f.write("\n")

    with open('testing.data', 'w') as f:
        for line in input_testing:
            for i in range(1,12):
                f.write(str(line[i])+",")
            f.write(str(line[12]))
            f.write("\n")
    #shuffle data


def func3a():
    input_list = []
    with open('training.data', 'r') as f:
        for line in f:
            currentline = line.split(",")
            currentline = list(map(int, currentline))
            input_list.append(currentline)

    training = np.array(input_list)
    training_x = training[:,:-1]
    training_y = training[:,-1:]


    input_testing = []
    with open('testing.data', 'r') as f:
        for line in f:
            currentline = line.split(",")
            currentline = list(map(int, currentline))
            input_testing.append(currentline)


    testing = np.array(input_testing)

    testing_x = testing[:,:-1]
    testing_y = testing[:,-1:]
    print(testing_y)
    rf = RandomForestClassifier()

    # train the model
    rf.fit(training_x, training_y)

    # predict the labels and report accuracy
    hard_pred = rf.predict(testing_x)
    testing_y = testing_y.flatten()
    print(hard_pred)
    print(testing_y)

    print(np.isclose(hard_pred, testing_y))

    acc = np.isclose(hard_pred, testing_y).sum() / len(hard_pred)
    print("Accuracy: {}".format(acc))

    # use predicted probabilities to construct ROC curve and AUC score
    soft_pred = rf.predict_proba(testing_x)
    fpr, tpr, thresh = roc_curve(testing_y, soft_pred[:, 1])
    auc = roc_auc_score(testing_y, soft_pred[:, 1])
    print("ROC Curve:")
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], "r--", alpha=.5)
    plt.show()
    print("AUC: {}".format(auc))
    x = f1_score(testing_y, hard_pred)
    print(x)



    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(training_x.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(training_x.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(training_x.shape[1]), indices)
    plt.xlim([-1, training_x.shape[1]])
    plt.show()


def func3c():
    input_list = []
    with open('training.data', 'r') as f:
        for line in f:
            currentline = line.split(",")
            currentline = list(map(int, currentline))
            input_list.append(currentline)

    training = np.array(input_list)
    training_x = training[:, :-1]

    training_x = np.delete(training_x, 2, 1)
    #training_x = np.delete(training_x, 1, 1)

    training_y = training[:, -1:]

    input_testing = []
    with open('testing.data', 'r') as f:
        for line in f:
            currentline = line.split(",")
            currentline = list(map(int, currentline))
            input_testing.append(currentline)

    testing = np.array(input_testing)

    testing_x = testing[:, :-1]
    testing_x = np.delete(testing_x, 2, 1)
    #testing_x = np.delete(testing_x, 1, 1)
    testing_y = testing[:, -1:]
    print(testing_y)
    rf = RandomForestClassifier()

    # train the model
    rf.fit(training_x, training_y)

    # predict the labels and report accuracy
    hard_pred = rf.predict(testing_x)
    testing_y = testing_y.flatten()
    print(hard_pred)
    print(testing_y)

    print(np.isclose(hard_pred, testing_y))

    acc = np.isclose(hard_pred, testing_y).sum() / len(hard_pred)
    print("Accuracy: {}".format(acc))

    # use predicted probabilities to construct ROC curve and AUC score
    soft_pred = rf.predict_proba(testing_x)
    fpr, tpr, thresh = roc_curve(testing_y, soft_pred[:, 1])
    auc = roc_auc_score(testing_y, soft_pred[:, 1])
    print("ROC Curve:")
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], "r--", alpha=.5)
    plt.show()
    print("AUC: {}".format(auc))
    x = f1_score(testing_y, hard_pred)
    print(x)

    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(training_x.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(training_x.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(training_x.shape[1]), indices)
    plt.xlim([-1, training_x.shape[1]])
    plt.show()

def func3d():
    input_list = []
    with open('training.data', 'r') as f:
        for line in f:
            currentline = line.split(",")
            currentline = list(map(int, currentline))
            input_list.append(currentline)

    training = np.array(input_list)
    training_x = training[:, :-1]

    #training_x = np.delete(training_x, 2, 1)
    # training_x = np.delete(training_x, 1, 1)

    training_y = training[:, -1:]

    input_testing = []
    with open('testing.data', 'r') as f:
        for line in f:
            currentline = line.split(",")
            currentline = list(map(int, currentline))
            input_testing.append(currentline)

    testing = np.array(input_testing)

    testing_x = testing[:, :-1]
    #testing_x = np.delete(testing_x, 2, 1)
    # testing_x = np.delete(testing_x, 1, 1)
    testing_y = testing[:, -1:]
    testing_y = testing_y.flatten()



    lr = LogisticRegression()

    # train the model
    lr.fit(training_x, training_y)

    # predict the labels and report accuracy
    hard_pred = lr.predict(testing_x)
    acc = np.isclose(hard_pred, testing_y).sum() / len(hard_pred)
    print("Accuracy: {}".format(acc))

    # use predicted probabilities to construct ROC curve and AUC score
    soft_pred = lr.predict_proba(testing_x)
    fpr, tpr, thresh = roc_curve(testing_y, soft_pred[:, 1])
    auc = roc_auc_score(testing_y, soft_pred[:, 1])
    print("ROC Curve:")
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], "r--", alpha=.5)
    plt.show()
    print("AUC: {}".format(auc))

    print(lr.coef_)

if __name__ == '__main__':
    #getTrainTest()
    #func3a()
    #func3c()
    func3d()