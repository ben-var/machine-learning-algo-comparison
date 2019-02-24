import numpy as np
import pandas as pd
import random as rand
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import scale, label_binarize
from sklearn import model_selection as ms
import csv
from time import time
from math import floor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC

''' gets feature names for each respective column '''
def get_bank_feature_names():
    return {0:"age",1:"job",2:"marital",3:"education",
    4:"default",5:"housing",6:"loan",7:"contact",8:"month",9:"day_of_week",
    10:"campaign",11:"pdays",12:"previous",13:"poutcome",14:"emp.var.rate",
    15:"cons.price.idx",16:"cons.conf.idx",17:"euribor3m",18:"nr.employed",
    19:"y"}

''' gets the feature type (i.e., numerical, categorical, or binary) '''
def bank_feature_type_indicies():
    return {"numeric":[0,10,11,12,14,15,16,17,18],
                    "categorical":[1,2,3,4,5,6,7,8,9,13],
                    "binary":[19]}

''' gets feature names for each respective column '''
def letter_feature_names():
    return {0:"x-box",1:"y-box",2:"width",3:"high",4:"onpix",5:"x-bar",
            6:"y-bar",7:"x2bar",8:"y2bar",9:"xybar",10:"x2ybr",11:"xy2br",
            12:"x-ege",13:"xegvy",14:"y-ege",15:"yegvx",16:"lettr"}

''' gets the feature type (i.e., numerical, categorical, or binary) '''
def letter_feature_type_indicies():
    return {"numeric":np.arange(16).tolist(),
            "categorical":[16]}

'''
    Turns string categorical information into numerical information
    params:
        df - dataframe to discretize
        index_list - indicies of df to discretize
        feat_names - a dictionary containing the index of the feats
                     as a key, and the name of the index as the value.
    returns:
        discretized dataframe
'''
def discretize_categories(df, index_list, feat_names):
    for i in index_list:
        feat = feat_names[i]
        categories = df[feat].unique()

        for j in range(len(categories)):
            df[feat][df[feat] == categories[j]] = j
    return df

''' helper method to numericalize the bank csv file input '''
def clean_parse_bank_file(data_file_path):
    bank_df = pd.read_csv(data_file_path)
    types = bank_feature_type_indicies()
    names = get_bank_feature_names()

    bank_df = discretize_categories(bank_df, types["categorical"], names)
    bank_df["y"][bank_df["y"] == "yes"] = 1
    bank_df["y"][bank_df["y"] == "no"] = 0

    if (data_file_path=="data/bank-additional-full.csv"):
        bank_df.to_csv("data/bank-additional-full-clean.csv", sep=",",
                                                                    index=False)
    elif (data_file_path=="data/bank-additional.csv"):
        bank_df.to_csv("data/bank-additional-clean.csv", sep=",", index=False)
    return bank_df.to_numpy()

''' helper method to numericalize the letter csv file input (labels) '''
def clean_parse_letter_file(data_file_path):
    letter_df = pd.read_csv(data_file_path)
    names = letter_feature_names()
    types = letter_feature_type_indicies()

    letter_df = discretize_categories(letter_df, types["categorical"], names)

    letter_df.to_csv("data/letter-recognition-clean.csv")
    return letter_df.to_numpy()

''' helper method to return np array from csv '''
def get_ndarray_from_csv(data_file_path):
    df = pd.read_csv(data_file_path)
    return df.to_numpy()

'''
    tests a single classifier across a single train test split
    arguments:
        clf - classifier to test
        X - feature data
        y - corresponding label
        test_size - portion of data to use as test data
        cm - Boolean for whether to print the confusion_matrix
'''
def test_single_classifier(clf, X, y, test_size=0.20, cm=False):
    X, X_t, y, y_t = ms.train_test_split(X, y, test_size=test_size)

    clf.fit(X, y)

    train_predict = clf.predict(X)
    train_acc = metrics.accuracy_score(y, train_predict)
    train_cm = metrics.confusion_matrix(y, train_predict)

    test_prediction = clf.predict(X_t)
    test_acc = metrics.accuracy_score(y_t, test_prediction)
    test_cm = metrics.confusion_matrix(y_t, test_prediction)

    print ("Train Accuracy = " + str(train_acc))
    if cm: print ("Train Confusion Matrix: \n" + str(train_cm))
    print ()
    print ("Test Accuracy = " + str(test_acc))
    if cm: print ("Test Confusion Matrix: \n" + str(test_cm))
    prec, rec, f1, _ = metrics.precision_recall_fscore_support(y_t, test_prediction)
    print ("Precision:\n" + str(prec))
    print ("Recall:\n" + str(rec))
    print ("F1 Score:\n" + str(f1))
    print()



'''
    generates a validation curve for a classifier given some data and a param
    arguments:
        clf - classifier to test
        clf_name - string name of the classifier
        X - feature data
        y - corresponding labels
        param_range - range of values to iterate the hyper-param over
        param_string - custom name of the parameter for formatting
        name_mod - additional string information to add to the graph
        ylim - custom ylimit for the graph
        cv - Determines the cross-validation splitting strategy. Possible inputs
        for cv are:

            None, to use the default 3-fold cross validation,
            integer, to specify the number of folds in a (Stratified)KFold,
            CV splitter,
            An iterable yielding (train, test) splits as arrays of indices.
        show - boolean whether to display the graph
        custom_param_range = custom range for the graph display
'''
def make_val_curve(clf, clf_name, X, y, param_range, param_name, param_string, name_mod=None, ylim=None, cv=None, show=False, custom_param_range=None):
    train_scores, test_scores = ms.validation_curve(
        clf, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring="accuracy", n_jobs=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve with " + clf_name)
    plt.xlabel(param_string)
    plt.ylabel("Score")
    if custom_param_range is not None:
        param_range=custom_param_range
    plt.xlim(param_range[0], param_range[-1])
    if ylim is not None:
        plt.ylim(*ylim)
    lw = 2
    plt.plot(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    if name_mod:
        param_name = param_name + name_mod
    plt.savefig('graphs/' + param_name + ' test.png')
    if show: plt.show()
    plt.close()
    return

'''
    generates a validation curve for an ensemble given some data and a param
    arguments:
        clf - classifier to test
        clf_name - string name of the classifier
        X - feature data
        y - corresponding labels
        param_range - range of values to iterate the hyper-param over
        name of the param to iterate over for the graph display
        name_mod - additional string information to add to the graph
        ylim - custom ylimit for the graph
        cv - Determines the cross-validation splitting strategy. Possible inputs
        for cv are:

            None, to use the default 3-fold cross validation,
            integer, to specify the number of folds in a (Stratified)KFold,
            CV splitter,
            An iterable yielding (train, test) splits as arrays of indices.
        show - boolean whether to display the graph
'''
def ensemble_val_curve(clf, clf_name, X, y, param_range, param_name, weak_param, cv=None, ylim=None, show=False):

    numeric_vals = param_range[0]
    weak_learners = param_range[1]
    train_scores, test_scores = ms.validation_curve(
        clf, X, y, param_name=param_name, param_range=weak_learners,
        cv=10, scoring="accuracy", n_jobs=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve with " + clf_name)
    plt.xlabel(weak_param)
    plt.ylabel("Score")
    plt.xlim(numeric_vals[0], numeric_vals[-1])
    if ylim is not None:
        plt.ylim(*ylim)
    lw = 2
    plt.plot(numeric_vals, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(numeric_vals, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.plot(numeric_vals, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(numeric_vals, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.savefig('graphs/' + weak_param + ' ensemble test.png')
    if show: plt.show()
    plt.close()
    return

''' generates a learning curve for a learner given some data '''
def plot_learning_curve(clf, clf_name, X, y, ylim=None, cv=None, show=False, name_mod=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    clf : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    clf_name : string
        For the title of the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title("Learning Curve with " + clf_name)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = ms.learning_curve(
        clf, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    if name_mod is not None:
        clf_name = clf_name + "_" + name_mod
    plt.savefig('graphs/' + clf_name + ' learn curve.png')
    if show: plt.show()
    plt.close()
    return

''' generates an ROC curve for a learner given some data and a learner
    arguments are similar to plot_learning_curve '''
def generate_ROC(X, y, clfs, clf_names, dataset_name, save_as="roc.png", test_size=0.20, show=False):

    X_train, X_test, y_train, y_test = ms.train_test_split(X, y,
                                                        test_size=test_size)

    plt.figure()

    for i in range(len(clfs)):
        clfs[i].fit(X_train, y_train)
        y_score = clfs[i].predict_proba(X_test)[:, 1]

        fpr, tpr, thres = metrics.roc_curve(y_test, y_score, pos_label=1)
        roc_auc = metrics.auc(fpr, tpr)
        lw=2
        plt.plot(fpr,tpr,lw=lw, label=clf_names[i] + ' AUC = %0.3f' % roc_auc)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) - ' + dataset_name)
    plt.legend(loc="lower right")
    plt.savefig(save_as)
    if show: plt.show()
    plt.close()

''' master method used to load a CSV file (handles both datasets of project) '''
def load_csv(data_file_path, class_index=-1, has_headers=False, scale_data=False, shuffle=False):

    """Load csv data in a numpy array.

    arguments:
        data_file_path (str): path to data file.

    Returns:
        features, labels as numpy arrays
    """
    is_bank = data_file_path=="data/bank-additional.csv" or \
              data_file_path=="data/bank-additional-full.csv" or \
              data_file_path=="data/bank-additional-full-clean.csv"

    is_letter = data_file_path=="data/letter-recognition.csv" or \
                data_file_path=="data/letter-regonition-clean.csv"

    if (is_bank):
        if data_file_path=="data/bank-additional-full-clean.csv":
            out = get_ndarray_from_csv(data_file_path)
        else:
            out = clean_parse_bank_file(data_file_path)

    elif (is_letter):
        if data_file_path=="data/letter-recognition-clean.csv":
            out = get_ndarray_from_csv(data_file_path)
        else:
            out = clean_parse_letter_file(data_file_path)

    else:
        handle = open(data_file_path, 'r')
        contents = handle.read()
        handle.close()
        rows = contents.split('\n')
        if (has_headers): rows = rows[1:]
        out = np.array([[float(i) for i in r.split(',')] for r in rows if r])

    if shuffle:
        indicies = np.arange(out.shape[0])
        np.random.shuffle(indicies)
        out = out[indicies]

    if class_index == -1:
        labels = out[:, class_index].astype('int')
        features = out[:, :class_index]

        if scale_data:
            features = scale(features)

        return features, labels

    elif class_index == 0:
        labels = out[:, class_index].astype('int')
        features = out[:, 1:]

        if scale_data:
            features = scale(features)

        return features, labels

    else:
        return out

''' returns a dict containing train/test times in seconds for each clf '''
def get_train_test_time(X, y, clfs, clf_names, test_size=0.20):
    X_train, X_test, y_train, y_test = ms.train_test_split(X, y,
                                                        test_size=test_size)
    times = {}
    for i in range(len(clfs)):
        name = clf_names[i]
        times[name] = {}
        t_start = time()
        clfs[i].fit(X_train, y_train)
        times[name]['train'] = time() - t_start
        q_start = time()
        clfs[i].predict(X_test)
        times[name]['query'] = time() - q_start

    return times

''' returns a dict containing error rates with cross-val in seconds for each clf '''
def get_error_rates(X, y, clfs, clf_names, cv=None, test_size=0.20):
    X_train, X_test, y_train, y_test = ms.train_test_split(X, y,
                                                        test_size=test_size)

    errors = {}
    for i in range(len(clfs)):
        name = clf_names[i]
        errors[name] = {}
        clfs[i].fit(X_train, y_train)

        train_predict = clfs[i].predict(X_train)
        train_acc = metrics.accuracy_score(y_train, train_predict)
        errors[name]['train'] = 1.0 - train_acc

        test_acc = np.mean(ms.cross_val_score(clfs[i], X, y, cv=cv))
        errors[name]['test'] = 1.0 - test_acc
    return errors

''' returns a list of tuned classifiers for a file. modify final params here '''
def get_tuned_clfs(csv_file):
    clfs = []
    clf_names = []

    if csv_file == "data/bank-additional-full-clean.csv":

        clfs.append(DecisionTreeClassifier(max_depth=5))
        clf_names.append('Decision Tree')

        clfs.append(KNeighborsClassifier(n_neighbors=30, metric='euclidean'))
        clf_names.append('KNN (euclidean distance)')

        clfs.append(SVC(kernel='rbf', gamma='auto', C=15, max_iter=200000, probability=True))
        clf_names.append('SVC (rbf kernel)')

        clfs.append(MLPClassifier(hidden_layer_sizes=(15, 2), max_iter=10000))
        clf_names.append('Neural Net')

        clfs.append(AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=50))
        clf_names.append('AdaBoost')

    elif csv_file == "data/letter-recognition-clean.csv":

        clfs.append(DecisionTreeClassifier())
        clf_names.append('Decision Tree')

        clfs.append(KNeighborsClassifier(n_neighbors=1, metric='euclidean'))
        clf_names.append('KNN')

        clfs.append(SVC(kernel='rbf', gamma='auto', C=5000, max_iter=200000, probability=True))
        clf_names.append('SVC (rbf kernel)')

        clfs.append(MLPClassifier(activation='relu', hidden_layer_sizes=(80,50,26), max_iter=10000))
        clf_names.append('Neural Net')

        clfs.append(AdaBoostClassifier(DecisionTreeClassifier(max_depth=20), n_estimators=100))
        clf_names.append('AdaBoost')

    return clfs, clf_names
