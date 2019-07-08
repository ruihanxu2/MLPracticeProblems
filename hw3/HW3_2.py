import numpy as np
from scipy import optimize
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
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




def getTrainTest():
    input = []
    with open('house_votes_84_test.csv', 'r') as f:
        for line in islice(f, 1, 51):
            currentline = line.split(",")
            currentline = list(map(int, currentline))
            input.append(currentline)



    test = np.array(input)
    test_y = test[:, :1]
    test_x = test[:, 1:]

    input = []
    with open('house_votes_84_train.csv', 'r') as f:
        for line in islice(f, 1, 386):
            currentline = line.split(",")
            currentline = list(map(int, currentline))
            input.append(currentline)



    train = np.array(input)
    train_y = train[:, :1]
    train_x = train[:, 1:]

    return train_x, train_y, test_x, test_y


def adaboost():


    train_x, train_y, test_x, test_y = getTrainTest()
    alpha = []
    h = []
    n_train = len(train_x)
    n_test = len(test_x)
    #weight vector
    d = np.ones(n_train) / n_train
    # Initialize weights
    for i in range(8):


        error = []
        for j in range(16):
            miss = []
            for k in range(len(train_x)):
                if train_x[k][j]!=train_y[k]:
                    miss.append(1)
                else:
                    miss.append(0)
            error.append(np.sum(d * miss))

        # now we have a error list that has 16 errors, choose the smallest one

        h_t = np.argmin(error)
        h.append(h_t)

        #now we get new d and alpha

        err_t = error[h_t]
        alpha_t = 0.5 * np.log((1 - err_t) / float(err_t))
        alpha.append(alpha_t)
        #get the error for h_t to update d
        error_t = []
        for k in range(len(train_x)):
            if train_x[k][h_t] != train_y[k]:
                error_t.append(-1)
            else:
                error_t.append(1)

        d = np.multiply(d, np.exp([float(x) * alpha_t for x in error_t]))
        d = d / np.sum(d)  # normalize
    pred = []

    for i in range(len(test_x)):
        res = 0
        for k in range(len(h)):
            res += alpha[k]*test_x[i][3]
        pred.append(res)

    pred = [1 if x > 0 else -1 for x in pred]
    true = 0
    false = 0
    for i in range(len(test_y)):
        if test_y[i] == pred[i]:
            true+=1
        else:
            false+=1

    acc = true/(true+false)

    print("accuarcy of adaboost is: ", acc)

class LogitReg():


    def __init__(self):
        self.lambda_ = []

    def fit(self, train_x, train_y):

        num =len(train_x[0])
        lambda_= np.zeros(num)

        def logLikelihood(lambda_, train_x, train_y):

            loss = [np.log(1 + np.exp(-y * np.sum(lambda_ * x))) for x, y in zip(train_x, train_y)]

            return np.sum(loss)

        res = optimize.minimize(logLikelihood, lambda_, args=(train_x, train_y))

        self.lambda_ = res.x

    def predict(self, test_x):
        pred = []
        for i in range(len(test_x)):
            temp = 1 / (1 + np.exp(-1 * np.sum(self.lambda_ * test_x[i])))
            if temp > 0.5:
                pred.append(1)
            else:
                pred.append(-1)
        return pred



if __name__ == '__main__':
    #adaboost()

    train_x, train_y, test_x, test_y = getTrainTest()
    lr = LogitReg()
    scores = cross_val_score(lr, train_x, train_y, cv=5, scoring="accuracy")
    print("Accuracy of Logistic Regression: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))