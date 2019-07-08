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

def perceptron(input_train, input_test):


    w = np.array([0,0,0], dtype=float)
    x = []
    y = []
    it = 0
    iter = []
    accuracy_train = []
    accuracy_test = []
    conv_train = []
    conv_test = []

    for p in input_train:
        x.append(np.array([p[0],p[1],p[2]], dtype=float))
        y.append(np.array([p[3]], dtype=float))
    while(1):
        iter.append(it)

        it += 1
        flag = True
        # get the accuracy at each update
        f_train = 0
        f_test = 0
        for i in range(0, len(input_train)):
            if y[i] * np.dot(w, x[i])<= 0:
                f_train += 1
        #misclassified = FN+FP/n
        accuracy_train.append(float(f_train) / float(len(input_train)))
        conv_train = np.diff(accuracy_train)


        for i in range(0, len(input_test)):
            if y[i] * np.dot(w, x[i])<= 0:
                f_test += 1
        accuracy_test.append(float(f_test) / float(len(input_test)))
        conv_test = np.diff(accuracy_test)

        #perceptron alg
        for i in range(0, len(input_train)):
            if y[i]* np.dot(w, x[i])<= 0:
                w = w+y[i]*x[i]
                flag = False
                break

        if flag:
            break
    # print("this is accuracy:")
    # print(len(accuracy_train))
    # print("this is iter:")
    # print(iter)


    plt.figure()
    plt.plot([time for time in iter[0::200]], [y_result for y_result in accuracy_train[0::200]])
    #plt.plot(iter, accuracy_test)
    plt.plot([time for time in iter[0::200]], [y_result for y_result in accuracy_test[0::200]])
    plt.show()


    return w
def func():

    #age, size
    input=[]
    with open('breast-cancer_cleaned.csv', 'r') as f:
        plots = csv.reader(f, delimiter = ',')
        next(f)
        for row in plots:
            numx = re.search('\d+', row[1])
            numy = re.search('\d+', row[2])


            if(row[0]) == "recurrence-events":
                input.append([int(numx.group()), int(numy.group()), 1 ,-1])

            else:
                input.append([int(numx.group()), int(numy.group()), 1 ,1])


    #shuffle data
    np.random.shuffle(input)



    # divide the data by test training
    input_train = []
    input_test = []

    r_age_train = []
    r_size_train  = []
    nr_age_train  = []
    nr_size_train  = []
    r_age_test  = []
    r_size_test  = []
    nr_age_test = []
    nr_size_test = []

    for points in input[0:int(0.7*len(input))]:
        input_train.append(points)
        if points[3] == -1:
            r_age_train.append(points[0])
            r_size_train.append(points[1])
        else:
            nr_age_train.append(points[0])
            nr_size_train.append(points[1])

    for points in input[int(0.7*len(input))+1:]:
        input_test.append(points)
        if points[3] == -1:
            r_age_test.append(points[0])
            r_size_test.append(points[1])
        else:
            nr_age_test.append(points[0])
            nr_size_test.append(points[1])


    plt.figure()
    plt.scatter(r_age_train, r_size_train, label='recurrence-events', color = 'k', marker='+', s = 50)
    plt.scatter(nr_age_train, nr_size_train, label='no-recurrence-events', color='k', marker='_', s=50)

    plt.xlabel('age')
    plt.ylabel('tumor-size')
    plt.title('Q2a: Projection of data')

    plt.legend().show()

    w1 = perceptron(input_train, input_test)
    x = np.linspace(30, 50, 20)
    plt.figure()
    plt.scatter(r_age_train, r_size_train, label='recurrence-events', color='k', marker='+', s=50)
    plt.scatter(nr_age_train, nr_size_train, label='no-recurrence-events', color='k', marker='_', s=50)
    if w1[1] != 0:
        plt.plot(x, (-w1[0]*x-w1[2])/w1[1])
    else:
        plt.vlines(x, -w1[2]/w1[0])
    plt.xlabel('age')
    plt.ylabel('tumor-size')
    plt.title('Q2b')
    plt.legend()
    plt.show()

    np.random.shuffle(input_train)

    w2 = perceptron(input_train, input_test)
    x = np.linspace(30, 50, 20)
    plt.figure()
    plt.scatter(r_age_train, r_size_train, label='recurrence-events', color='k', marker='+', s=50)
    plt.scatter(nr_age_train, nr_size_train, label='no-recurrence-events', color='k', marker='_', s=50)
    if w1[1] != 0:
        plt.plot(x, (-w1[0]*x-w1[2])/w1[1])
    else:
        plt.vlines(x, -w1[2]/w1[0])
    if w1[1] != 0:
        plt.plot(x, (-w2[0]*x-w2[2])/w2[1])
    else:
        plt.vlines(x, -w2[2]/w2[0])
    plt.xlabel('age')
    plt.ylabel('tumor-size')
    plt.title('Q2d')
    plt.legend()
    plt.show()



def func3a():
    # read data and take 25% of it

    input_list = []
    with open('covtype_small.data', 'r') as f:
        for line in f:
            currentline = line.split(",")
            currentline = list(map(int, currentline))
            input_list.append(currentline)


    print(len(input_list))

    input = np.array(input_list)
    x = input[:,:-1]
    y = input[:,-1]

    kfold = KFold(n_splits=10, random_state=None)

    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')

    for train_index, test_index in kfold.split(x, y):

        train1_x = [x[i] for i in train_index]
        train1_y = [y[i] for i in train_index]

        test1_x = [x[i] for i in test_index]
        test1_y = [y[i] for i in test_index]


        clf = OneVsRestClassifier(LogisticRegression(C = 10000, solver = "liblinear", max_iter= 4000))
        y_score = clf.fit(train1_x, train1_y).predict(test1_x)

        y_score = label_binarize(y_score, classes = [1,2,3,4,5,6,7])
        test1_y = label_binarize(test1_y, classes = [1,2,3,4,5,6,7])

        n_classes = y_score.shape[1]

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):

            fpr[i], tpr[i], _ = roc_curve(test1_y[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(test1_y.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        print("micro is ",roc_auc["micro"])
        # Compute macro-average ROC curve and ROC area

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves

        # plt.plot(fpr["micro"], tpr["micro"],
        #          label='micro-average ROC curve (area = {0:0.2f})'
        #                ''.format(roc_auc["micro"]),
        #          color='deeppink', linestyle=':', linewidth=4)


        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]), linestyle=':', linewidth=1)


        # for i, color in zip(range(n_classes), colors):
        #     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
        #              label='ROC curve of class {0} (area = {1:0.2f})'
        #                    ''.format(i, roc_auc[i]))

    plt.legend(loc="lower right")
    plt.show()
def func3b():

    input_list=[]
    with open('covtype_small.data', 'r') as f:
        for line in f:
            currentline = line.split(",")
            currentline = list(map(int, currentline))
            input_list.append(currentline)
        print(len(line))

    print(len(input_list))


    input = np.array(input_list)

    x = input[:, :-1]
    y = input[:, -1]


    K = [1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]

    print(len(x))
    print(len(y))

    test_x = x[int(0.9*len(x)):]
    test_y = y[int(0.9*len(y)):]

    x = x[:int(0.9*len(x))]
    y = y[:int(0.9*len(y))]


    kfold = KFold(n_splits=9, random_state=None)

    acc_mat = [[],[],[],[],[],[],[],[],[],[],[],[]]

    for train_index, valid_index in kfold.split(x, y):
        print("start")
        train_x = [x[i] for i in train_index]
        train_y = [y[i] for i in train_index]

        valid_x = [x[i] for i in valid_index]
        valid_y = [y[i] for i in valid_index]

        for i in range(12):
            print(i);
            c = 1/K[i]
            clf = OneVsRestClassifier(LogisticRegression(C=c, solver="liblinear"))
            pred_valid_y = clf.fit(train_x, train_y).predict(valid_x)

            acc = np.isclose(pred_valid_y, valid_y).sum() / len(pred_valid_y)
            acc_mat[i].append(acc)


            #print("Accuracy: {}".format(acc))

    acc_mat = np.array(acc_mat)
    for i in range(12):
        print("mean validation accuracy for", K[i]," is ", np.mean(acc_mat[i]))
def func3c():

    input_list=[]
    with open('covtype_small.data', 'r') as f:
        for line in f:
            currentline = line.split(",")
            currentline = list(map(int, currentline))
            input_list.append(currentline)

    print(len(input_list))


    input = np.array(input_list)

    x = input[:, :-1]
    y = input[:, -1]


    #the test case I set aside from 3b
    test_x = x[int(0.9*len(x)):]
    test_y = y[int(0.9*len(y)):]

    train_x = x[:int(0.9*len(x))]
    train_y = y[:int(0.9*len(y))]


    K = [1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]

    acc = []

    for i in range(12):
        print(i);
        c = 1 / K[i]
        clf = OneVsRestClassifier(LogisticRegression(C=c, solver="liblinear"))
        pred_valid_y = clf.fit(train_x, train_y).predict(test_x)

        acc.append(np.isclose(pred_valid_y, test_y).sum() / len(pred_valid_y))

    for i in range(12):
        print("Test set accuracy for", K[i]," is ", acc[i])
def func3d():

    input_list=[]
    with open('covtype_small.data', 'r') as f:
        for line in f:
            currentline = line.split(",")
            currentline = list(map(int, currentline))
            input_list.append(currentline)

    print(len(input_list))


    input = np.array(input_list)

    x = input[:, :-1]
    y = input[:, -1]


    #the test case I set aside from 3b



    K = [1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]

    acc = []
    auc_plt = []
    for i in range(12):
        test_x = x[int(0.9 * len(x)):]
        test_y = y[int(0.9 * len(y)):]

        train_x = x[:int(0.9 * len(x))]
        train_y = y[:int(0.9 * len(y))]
        print(i);
        c = 1 / K[i]
        clf = OneVsRestClassifier(LogisticRegression(C=c, solver="liblinear"))
        pred_valid_y = clf.fit(train_x, train_y).predict(test_x)

        acc.append(np.isclose(pred_valid_y, test_y).sum() / len(pred_valid_y))

        # compute auc
        pred_valid_y = label_binarize(pred_valid_y, classes=[1, 2, 3, 4, 5, 6, 7])
        test_y = label_binarize(test_y, classes=[1, 2, 3, 4, 5, 6, 7])

        n_classes = pred_valid_y.shape[1]

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(test_y[:, i], pred_valid_y[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(test_y.ravel(), pred_valid_y.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        auc_plt.append(roc_auc["micro"])



    print(len(auc_plt))
    print(len(acc))

    plt.figure()
    plt.boxplot(acc)
    plt.ylabel('accuracy')
    plt.show()

    plt.figure()
    plt.boxplot(auc_plt)
    plt.ylabel('auc')
    plt.show()
def func3e():

    input_list=[]
    with open('covtype_small.data', 'r') as f:
        for line in f:
            currentline = line.split(",")
            currentline = list(map(int, currentline))
            input_list.append(currentline)

    print(len(input_list))


    input = np.array(input_list)

    x = input[:, :-1]
    y = input[:, -1]


    #the test case I set aside from 3b



    K = [1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]

    acc = []
    auc_plt = []

    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC with different K')
    for i in range(12):
        test_x = x[int(0.9 * len(x)):]
        test_y = y[int(0.9 * len(y)):]

        train_x = x[:int(0.9 * len(x))]
        train_y = y[:int(0.9 * len(y))]
        print(i);
        c = 1 / K[i]


        clf = OneVsRestClassifier(LogisticRegression(C=c, solver="liblinear"))
        pred_valid_y = clf.fit(train_x, train_y).predict(test_x)

        acc.append(np.isclose(pred_valid_y, test_y).sum() / len(pred_valid_y))

        # compute auc
        pred_valid_y = label_binarize(pred_valid_y, classes=[1, 2, 3, 4, 5, 6, 7])
        test_y = label_binarize(test_y, classes=[1, 2, 3, 4, 5, 6, 7])

        n_classes = pred_valid_y.shape[1]

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(test_y[:, i], pred_valid_y[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(test_y.ravel(), pred_valid_y.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        print("micro is ", roc_auc["micro"])
        # Compute macro-average ROC curve and ROC area

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves

        # plt.plot(fpr["micro"], tpr["micro"],
        #          label='micro-average ROC curve (area = {0:0.2f})'
        #                ''.format(roc_auc["micro"]),
        #          color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label=('K = '+str(K[i])+'macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"])), linestyle=':', linewidth=1)
    plt.show()



if __name__ == '__main__':
    #func()
    #func3a()

    #func3b()
    #func3c()
    #func3d()
    func3e()