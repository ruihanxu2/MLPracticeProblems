import numpy as np
from matplotlib import pyplot as plt
import csv
import re
from itertools import cycle
from random import choices
from scipy import interp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, auc

if __name__ == '__main__':
    # func()
    # func3a()
    input_all = []
    with open('covtype.data', 'r') as f:
        for line in f:
            currentline = line.split(",")
            currentline = list(map(int, currentline))
            input_all.append(currentline)

    print(len(input_all))
    input_list = choices(input_all, k=len(input_all) // 4)
    print(len(input_list))

    with open('covtype_small.data', 'w') as f:
        for line in input_list:
            for i in range(54):
                f.write(str(line[i])+",")
            f.write(str(line[54]))
            f.write("\n")
