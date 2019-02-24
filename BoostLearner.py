from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import numpy as np
import util as u
import time

start = time.time()

print ("Running AdaBoost script...")

''' TOGGLE THESE TWO LINES FOR THE TWO DIFFERENT FILES '''
csv_file = "data/letter-recognition-clean.csv"
# csv_file = "data/bank-additional-full-clean.csv"
X, y = u.load_csv(csv_file, shuffle=True, has_headers=True)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

if csv_file == "data/bank-additional-full-clean.csv":
    ###### TUNING HYPER PARAMS ######
    name = "AdaBoost - Bank"
    cv = ShuffleSplit(n_splits=10, test_size=0.2)

    clf = AdaBoostClassifier(DecisionTreeClassifier())
    drange = np.arange(1,10)
    trees = []
    for i in drange:
        trees.append(DecisionTreeClassifier(max_depth=i))
    param_range = (drange, trees)
    u.ensemble_val_curve(clf, name, X, y, param_range=param_range, cv=cv,
                    param_name="base_estimator", weak_param="Max Depth", show=False)

    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2))
    lrange = np.arange(100,300)
    trees = []
    for i in lrange:
        trees.append(DecisionTreeClassifier(max_leaf_nodes=i))
    param_range = (lrange, trees)
    u.ensemble_val_curve(clf, name, X, y, param_range=param_range, cv=cv,
                param_name="base_estimator", weak_param="Max Leaf Nodes", show=False)

    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1))
    erange = np.arange(50,101)
    u.make_val_curve(clf, name, X, y, param_range=erange, ylim=None,
                        name_mod='1depthbank', param_name="n_estimators",
                        param_string="# of estimators", cv=cv,
                        show=False)
    ###### GENERATING LEARNING CURVE ######
    test_size = 0.20
    name = "AdaBoost - Bank"
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=50)
    u.plot_learning_curve(clf, name, X, y, cv=cv, n_jobs=None, show=False,
                            ylim=(0,1.10), train_sizes=np.linspace(.1, 1.0, 5)+

    ##### TESTING CLASSIFIER AGAINST BENCHMARK #####
    print("Benchmark Default AdaBoost Test")
    clf = AdaBoostClassifier(DecisionTreeClassifier())
    u.test_single_classifier(clf, X, y, test_size=test_size, cm=True)

    print ("Tuned AdaBoost Test - Single Iteration")
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=50)
    u.test_single_classifier(clf, X, y, test_size=test_size, cm=True)


elif csv_file == 'data/letter-recognition-clean.csv':
    ###### TUNING HYPER PARAMS ######
    name = "AdaBoost - Letter"
    cv = ShuffleSplit(n_splits=5, test_size=0.2)

    clf = AdaBoostClassifier(DecisionTreeClassifier())
    drange = np.arange(1,30)
    trees = []
    for i in drange:
        trees.append(DecisionTreeClassifier(max_depth=i))
    param_range = (drange, trees)
    u.ensemble_val_curve(clf, name, X, y, param_range=param_range, cv=cv,
                    param_name="base_estimator", weak_param="Max Depth", show=False)

    clf = AdaBoostClassifier(DecisionTreeClassifier())
    lrange = np.arange(100,150)
    trees = []
    for i in lrange:
        trees.append(DecisionTreeClassifier(max_leaf_nodes=i, max_depth=20))
    param_range = (lrange, trees)
    u.ensemble_val_curve(clf, name, X, y, param_range=param_range, cv=cv,
                param_name="base_estimator", weak_param="Max Leaf Nodes",
                ylim=None, show=False)

    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=22))
    erange = np.arange(100,126)
    u.make_val_curve(clf, name, X, y, param_range=erange, ylim=None,
                        name_mod='20depthletter', param_name="n_estimators",
                        param_string="# of estimators", cv=cv,
                        show=False)

    ###### GENERATING LEARNING CURVE ######
    test_size = 0.20
    clf = AdaBoostClassifier(
            DecisionTreeClassifier(max_depth=20), n_estimators=100)
    u.plot_learning_curve(clf, name, X, y, cv=cv, show=False, ylim=(0,1.1),
                                            train_sizes=np.linspace(.1, 1.0, 10))

    ##### TESTING CLASSIFIER AGAINST BENCHMARK #####
    print("Benchmark Default AdaBoost Test")
    clf = AdaBoostClassifier(DecisionTreeClassifier())
    u.test_single_classifier(clf, X, y, test_size=test_size, cm=True)

    print ("Tuned AdaBoost Test - Single Iteration")
    clf = AdaBoostClassifier(
            DecisionTreeClassifier(max_depth=20), n_estimators=100)
    u.test_single_classifier(clf, X, y, test_size=test_size, cm=True)

print ('Time to run: ' + str(time.time() - start) + ' seconds.')
