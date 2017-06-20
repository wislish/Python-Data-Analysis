import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import pandas as pd
import time
from datetime import datetime
from datetime import timedelta
import numpy as np
import itertools
import os
import collections

from sklearn import tree
from sklearn.cluster import KMeans

from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

from sklearn import preprocessing

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

sns.set_style('whitegrid',{'font.sans-serif':['simhei','Arial']})

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        os.subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")


def classify(X,y, clf,**para):
    # y = profile["Loss"].as_matrix()
    # X = profile[features].as_matrix()

    kf = KFold(n_splits=10)
    skf = StratifiedKFold(n_splits=6)

    # print(**para)
    classifier = clf(**para)
    name = str(classifier).split("(")[0]


    # dt = tree.DecisionTreeClassifier(min_samples_split=min_split, max_depth=max_dep)
    print("{0} has been established with {1}".format(name, para))
    # lr = LogisticRegression(penalty='l1')

    for train_index, test_index in skf.split(X, y):
        #     print("TRAIN:",train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        print("10-fold Score is: {0}".format(score))

    return classifier,y_test, y_pred

def standarization(X):
    x_scaler = preprocessing.StandardScaler().fit(X)

    return x_scaler

def preProcess(profile):
    profile.dropna(axis=0, how='any', inplace=True)

    # return profile
def clustering(X,y,clf,**para):

    skf = StratifiedKFold(n_splits=6)
    cluster = clf(**para)

    ## shuffle the data before clustering
    r = np.random.permutation(len(y))
    X = X[r,:]
    y= y[r]

    cluster.fit(X)
    name = str(cluster).split("(")[0]

    print("{0} has been established with {1}".format(name, para))

    for train_index, test_index in skf.split(X, y):
        #     print("TRAIN:",train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        cluster.fit(X_train, y_train)
        y_pred = cluster.predict(X_test)
        score = max(accuracy_score(y_test, y_pred),1-accuracy_score(y_test, y_pred))
        print("10-fold Score is: {0}".format(score))

    return cluster,y_test, y_pred


if __name__ == "__main__":
    profile = pd.read_csv("/home/maoan/maidianAnalysis/level2-uianalysis/用户画像.csv")

    ## Construct the features name.
    features = profile.columns[1:2].tolist() + profile.columns[4:].tolist()

    ## preProcess
    print(profile.shape)
    preProcess(profile)

    X = profile[features].as_matrix()
    y = profile["Loss"].as_matrix()
    ## choose the classifier
    min_split = 20
    max_dep = 3
    dt = tree.DecisionTreeClassifier
    lr = LogisticRegression

    ## conduct the classfiy and pass the related parameters
    # _, y_test, y_pred = classify(X,y=y,clf=dt,
    #                              min_samples_split=min_split, max_depth=max_dep)

    ##########################################################
    ##########################################################
    ## conduct the clustering and pass the related parameters#
    kmeans = KMeans
    num_clusters = 2

    ## Done standarization before if you want to use K-Means
    scaler = standarization(X)
    res, y_test, y_pred = clustering(scaler.transform(X),y,clf=kmeans, n_clusters=num_clusters)

    print("==========================")
    print("Clusters centroids are:")
    print('{:15} | {:^9} | {:^9}'.format('', 'Cluster0', 'Cluster1'))
    fmt = '{:15} | {:9.4f} | {:9.4f}'
    # print(features)
    origin_x = scaler.inverse_transform(res.cluster_centers_)
    for i in range(len(features)):

        print(fmt.format(features[i],origin_x[0][i],origin_x[1][i]))
        # print(res.cluster_centers_[i])
        # print(scaler.inverse_transform(res.cluster_centers_[i]))

    class_names = ['Leave', 'Stay']

    print("==========================")
    print("The true positive feature is '{0}'".format(class_names[1]))
    print("The precision score is {0}".format(precision_score(y_true=y_test,y_pred=y_pred)))
    print("The recall score is {0}".format(recall_score(y_true=y_test,y_pred=y_pred)))
    print("The F1 score is {0}".format(f1_score(y_true=y_test,y_pred=y_pred)))
    print("==========================")

    ## Visualize the result
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')
    # plt.show()

