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
from matplotlib import cm

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import subprocess
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn import metrics

from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
from collections import Counter

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
    
    :param cm: 混淆矩阵, confusion matrix
    :param classes: 两个类别的名字
    :param normalize: 是否正则化
    :param title: 图的名称
    :param cmap: not known
    :return: 
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
    plt.show()

def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.

    Args
    ----
    :param tree -- scikit-learn DecsisionTree.
    :param feature_names -- list of feature names.
    """
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")

def visualize_silhouette_score(X,y_km):
    """
    视觉化 silhouette score.
    :param X: 训练数据，矩阵形式
    :param y_km: 标记数据，列表形式
    :return: 
    """

    cluster_labels = np.unique(y_km)
    n_clusters = cluster_labels.shape[0]
    silhouette_vals = metrics.silhouette_samples(X,
                                         y_km,
                                         metric='euclidean')
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []
    for i, c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[y_km == c]
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color = cm.jet(i / n_clusters)
        plt.barh(range(y_ax_lower, y_ax_upper),
                c_silhouette_vals,
                height=1.0,
                edgecolor='none',
                color=color)
        yticks.append((y_ax_lower + y_ax_upper) / 2)
        y_ax_lower += len(c_silhouette_vals)

    silhouette_avg = np.mean(silhouette_vals)
    plt.axvline(silhouette_avg,
                color="red",
                linestyle="--")
    plt.yticks(yticks, cluster_labels + 1)
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette coefficient')
    plt.show()

def stratifiedCV(X, y, n_splits = 6):
    """
    stratified 交叉验证
    :param X: 训练数据，矩阵
    :param y: 标记数据。矩阵
    :param n_splits: 分成的份数
    :return: 生成器，返回cv中每一份的数据
    """

    skf = StratifiedKFold(n_splits=n_splits)

    for train_index, test_index in skf.split(X, y):
        #     print("TRAIN:",train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        yield X_train, y_train, X_test, y_test

def regulCV(X,y,n_splits = 10):
    """
        cross-validation 交叉验证
    :param X: 训练数据，矩阵
    :param y: 标记数据。矩阵
    :param n_splits: 分成的份数
    :return: 生成器，返回cv中每一份的数据 
    """
    kf = KFold(n_splits=n_splits)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        yield X_train, y_train, X_test, y_test

def standarization(profile,features=None):
    """
    正则化数据，如果有输入特征，则只正则化这些特征的值。
    否则，正则化所有的数值型特征。
    :param profile: 原始用户特征DataFrame
    :param features: 需要正则化的特征集
    :return: 正则化之后的用户特征DataFrame,以及原来的初始DataFrame
    """

    if features:
        select_features = features
    else:
        select_features = profile.select_dtypes(include=[np.number]).columns

    norm_profile = profile.copy()
    norm_features_df = norm_profile[select_features]
    scaler = preprocessing.StandardScaler().fit(norm_features_df.values)
    norm_values = scaler.transform(norm_features_df.values)
    norm_profile[select_features] = norm_values



    return norm_profile,profile

def confusionMatrixAnalysis(class_names, y_test,y_pred):
    """
    测算混淆矩阵的各个指标
    :param class_names:类别的名称 
    :param y_test: 真实标记数据值
    :param y_pred: 预测标记数据值
    :return: 混淆矩阵的图
    """

    print("========Confusion Matrix Analysis==========")
    print("The true positive feature is '{0}'".format(class_names[1]))
    print("The precision score is {0}".format(precision_score(y_true=y_test, y_pred=y_pred)))
    print("The recall score is {0}".format(recall_score(y_true=y_test, y_pred=y_pred)))
    print("The F1 score is {0}".format(f1_score(y_true=y_test, y_pred=y_pred)))
    print("==========================")

    ## Visualize the result
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')
    plt.show()

def decisionTreeClassify(X,y, features,class_names, straitified=True,**para):
    """
    决策树分类器
    :param X: 训练数据，矩阵
    :param y: 标记数据，矩阵
    :param features: 特征的名称，按顺序
    :param class_names: 类别的名称
    :param straitified: 是否采用stratified交叉验证
    :param para: 其他　分类器所需要的参数
    :return: 决策树
    """

    dt = tree.DecisionTreeClassifier(**para)

    print("{0} has been established with {1}".format("Decision Tree Classifier ", para))

    cv = stratifiedCV(X,y) if straitified else regulCV(X,y)

    for X_train, y_train, X_test, y_test in cv:
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        print("{0} Score is: {1}".format("Straitified Cross Validation",score))

    confusionMatrixAnalysis(class_names,y_test,y_pred)

    visualize_tree(dt,features)

    return dt

def randomForestClassify(X,y,features,class_names, straitified=True,**para):
    """
    随机森林分类器
    :param X: 训练数据，矩阵
    :param y: 标记数据，矩阵
    :param features: 特征的名称，按顺序
    :param class_names: 类别的名称
    :param straitified: 是否采用stratified交叉验证
    :param para: 其他　分类器所需要的参数
    :return: 随机森林
    """
    rf = RandomForestClassifier(**para)

    print("{0} has been established with {1}".format("Random Forest Classifier ", para))


    cv = stratifiedCV(X,y) if straitified else regulCV(X,y)

    for X_train, y_train, X_test, y_test in cv:
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        print("{0} Score is: {1}".format("Straitified Cross Validation",score))

    confusionMatrixAnalysis(class_names,y_test,y_pred)

    ## feature importance analysis

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\n========Feature Importance==========\n")

    for f in range(X_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30,
                                features[indices[f]],
                                importances[indices[f]]))

    # y_pred = rf.predict(test_x)
    # score = accuracy_score(test_y, y_pred)
    # print("{0} Score is: {1}".format("Validation Score is ", score))
    # confusionMatrixAnalysis(class_names,test_y,y_pred)

    return rf

def logiRegressionClassify(X,y, features,class_names, straitified=True,**para):
    """
    逻辑回归分类器
    :param X: 训练数据，矩阵
    :param y: 标记数据，矩阵
    :param features: 特征的名称，按顺序
    :param class_names: 类别的名称
    :param straitified: 是否采用stratified交叉验证
    :param para: 其他　分类器所需要的参数
    :return: 逻辑回归分类器
    """
    lr = LogisticRegression(**para)

    print("==================================")
    print("{0} has been established with {1}".format("Decision Tree Classifier ", para))

    cv = stratifiedCV(X, y) if straitified else regulCV(X, y)

    for X_train, y_train, X_test, y_test in cv:
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        print("{0} Score is: {1}".format("Straitified Cross Validation", score))

    confusionMatrixAnalysis(class_names, y_test, y_pred)

    print("\n========Coefficients==========\n")

    indices = np.argsort(abs(lr.coef_[0]))[::-1]

    for f in range(X_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30,
                                features[indices[f]],
                                lr.coef_[0][indices[f]]))

    return lr

def preProcess(profile):
    profile.dropna(axis=0, how='any', inplace=True)
    profile = transfromFeatures(profile)

    return profile

def transfromFeatures(predDF):
    """
    把名义性或者字符型特征转化成数值型特征，并补全缺失值。
    :param predDF: 需要转换的数据框，DataFrame格式
    :return: 全新的数据框，DataFrame
    """

    print("Transform the categorical features into numerical.")
    cat_col = predDF.select_dtypes(exclude=[np.number]).columns
    num_col = predDF.select_dtypes(include=[np.number]).columns


    num_col_hosts = predDF[num_col].apply(lambda x: x.fillna(x.mean()))
    cat_col_hosts = predDF[cat_col].apply(lambda x: x.fillna(x.value_counts().index[0]))
    print('numerical column null count:' + (str)(num_col_hosts.isnull().sum().sum()))
    print('categorical column null count:' + (str)(cat_col_hosts.isnull().sum().sum()))

    le = preprocessing.LabelEncoder()
    coding_hosts = cat_col_hosts.apply(lambda x: le.fit_transform(x))

    ## merge two dataframes
    frames = [coding_hosts, num_col_hosts]
    cocat_hosts = pd.concat(frames, axis=1)

    col = cocat_hosts.columns.tolist()
    # print(col)
    np.random.shuffle(col)

    recurring_hosts = cocat_hosts[col].copy(deep=True)

    return recurring_hosts

def clustering(X,clf,**para):
    """
    聚类
    :param X: 训练数据，矩阵
    :param clf: 聚类器
    :param para: 聚类器所需要的参数
    :return: 训练过后的聚类器
    """
    # skf = StratifiedKFold(n_splits=3)
    cluster = clf(**para)

    ## shuffle the data before clustering
    r = np.random.permutation(X.shape[0])
    X = X[r,:]
    # y= y[r]

    cluster.fit_predict(X)
    name = str(cluster).split("(")[0]

    print("{0} has been established with {1}".format(name, para))


    # score = max(accuracy_score(y_test, y_pred),1-accuracy_score(y_test, y_pred))

    """
    ---R-SCORE (adjusted rand score)---
    Random (uniform) label assignments have a ARI score close to 0.0 
        for any value of n_clusters and n_samples (which is not the case for raw Rand index or the V-measure for instance).
    Bounded range [-1, 1]: 
        negative values are bad (independent labelings), 
        similar clusterings have a positive ARI, 1.0 is the perfect match score.
    No assumption is made on the cluster structure: 
        can be used to compare clustering algorithms such as k-means 
        which assumes isotropic blob shapes with results of spectral clustering algorithms 
        which can find cluster with “folded” shapes.
    """
    # r_score = metrics.adjusted_rand_score(y_test, y_pred)

    """
    ---NMI-SCORE Normalized Mutual Information)---
    
    Random (uniform) label assignments have a ARI score close to 0.0 
        for any value of n_clusters and n_samples (which is not the case for raw Rand index or the V-measure for instance).
    Bounded range [-1, 1]: 
        Values close to zero indicate two label assignments that are largely independent, 
        while values close to one indicate significant agreement. 
        Further, values of exactly 0 indicate purely independent label assignments and a AMI of exactly 1 
        indicates that the two label assignments are equal (with or without permutation).
    No assumption is made on the cluster structure: 
        can be used to compare clustering algorithms such as k-means 
        which assumes isotropic blob shapes with results of spectral clustering algorithms 
        which can find cluster with “folded” shapes.
    """
    # nmi_score = metrics.adjusted_mutual_info_score(y_test, y_pred)

    # print("10-fold accuracy is: {0}, "
    #       "the Adjusted R-score is: {1}, "
    #       "the NMI-SCORE is: {2}".format(score,r_score,nmi_score))

    return cluster

def classifyUsers(profile,class_names = ["Negative","Positive"],stand_col_names=None,im_balance=False):
    """
    根据输入的用户特征，来进行分类。其中，数据集中最后一列为需要预测的特征，会自动进行正则化以及
    丢掉缺失数据。还可以对不平衡的数据集进行插补。
    
    目前默认采用三种分类算法，分别为决策树，随机森林，逻辑回归。
    :param profile_path: 存储用户特征文件的路径，其中最后一个特征为预测特征
    :param class_names: 类别名称
    :param stand_col_names: 需要正则化的特征名，如果无，则自动正则化所有数值型特征
    :param im_balance: 是否需要采用SMOT-Boderline使得两类数据变得均衡
    :return: 
    """

    # profile = pd.read_csv("/home/maoan/maidianAnalysis/level3-growth/userProfile.csv")
    # profile = pd.read_csv("/home/maoan/maidianAnalysis/level3-growth/user_actions.csv")
    # profile = pd.read_csv(profile_path)

    ## Construct the features name.
    features = profile.columns.tolist()[:-1]
    pred_feature= profile.columns.tolist()[-1]


    ## basic data preprocessing
    profile.dropna(axis=0, how='any', inplace=True)
    # col_names = ['Freq','BattleRatio']
    # profile = transfromFeatures(profile)

    # standarlization
    norm_profile, _ = standarization(profile,stand_col_names)

    print("Features are: {0}".format(features))

    X = profile[features].as_matrix()
    y = profile[pred_feature].as_matrix()

    norm_x = norm_profile[features].as_matrix()
    norm_y = norm_profile[pred_feature].as_matrix()


    # dealing with the imbalanced data problem
    if im_balance:
        print('Original dataset shape {}'.format(Counter(norm_y)))
        print(np.median(norm_x, axis=0))
        sm = SMOTE(random_state=42,kind="borderline2")
        X_res, y_res = sm.fit_sample(norm_x, norm_y)
        X,y = X_res, y_res
        norm_x,norm_y = X_res, y_res

    # class_names = ['Non-VIP', 'VIP']

    ## choose the classifier and set the parameters
    min_split = 20
    max_dep = 3

    decisionTreeClassify(X,y,features,class_names,min_samples_split=min_split, max_depth=max_dep)

    # use normalized data
    logiRegressionClassify(norm_x,norm_y,features,class_names,penalty="l1")

    randomForestClassify(X,y,features,class_names,n_estimators=250)


def clusterUsers(profile_df):

    """
    对用户进行聚类。可以在里面选择需要的聚类模型，输入包括所有训练特征，不含要预测的特征。
    以后可以考虑分离各个聚类模型。
    :param profile_df: 用户特征的数据框, DataFrame格式
    :return: 
    """
    # profile = pd.read_csv(profile_file)

    ## Construct the features name.
    # features = profile.columns[1:2].tolist() + profile.columns[4:].tolist()
    # ## preProcess
    # print(profile.shape)
    # preProcess(profile)

    select_fea = profile_df.columns.tolist()

    profile_df.dropna(axis=0, how='any', inplace=True)

    # profile_df = transfromFeatures(profile_df)

    ## select the clustering methods and set the parameters
    kmeans = KMeans
    amcluster = AgglomerativeClustering
    max_clusters = 3

    ## Done standarization before if you want to use K-Means
    X=profile_df.as_matrix()
    scaler = preprocessing.StandardScaler().fit(X)

    # stand_features = ["Freq","BattleRatio"]


    ##plot the sum of within cluster errors as the num of clusters grow...
    distortions = []
    for i in range(2,max_clusters+1):

        print("==========================")
        res = clustering(scaler.transform(X),clf=kmeans, n_clusters=i)
        distortions.append(res.inertia_)
        origin_x = scaler.inverse_transform(res.cluster_centers_)

        print("Clusters centroids are:")
        head_row = ('{:15}' + ''.join([' {:^9} |'] * len(select_fea))).format('',*select_fea)
        print(head_row)

        fmt = '{:15} |' + ''.join([' {:^9.4} |'] * len(select_fea))
        for num in range(i):
            name_row = "Cluster" + str(num+1)
            print(fmt.format(name_row, *(origin_x[num].tolist())))

        visualize_silhouette_score(scaler.transform(X),res.labels_)

    plt.plot(range(2, max_clusters+1), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.show()


if __name__ == "__main__":
    user_actions= pd.read_csv("./user_actions.csv")
    user_profile = pd.read_csv("./userFile_vip.csv")

    class_names = ['Non-VIP', 'VIP']

    # classifyUsers(user_actions,class_names)
    clusterUsers(user_profile)