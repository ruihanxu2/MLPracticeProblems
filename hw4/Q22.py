
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





def plotit():

    X1 = np.linspace(-10000, 10000, 200)
    X2 = np.linspace(-10000, 10000, 200)
    fx = []
    gx = []
    Zx = []
    for x1 in X1:
        for x2 in X2:
            fx.append(x1**2+x2**2)
            gx.append(1-x1-x2)


    print(len(fx))
    print(len(gx))

    plt.scatter(gx, fx)

    plt.show()


if __name__ == '__main__':
    plotit()





    # SUPPORT VECTOR MACHINE
