from sklearn import neighbors, metrics
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import util as u
import time


start = time.time()
print ("Running kNN script ...")

''' TOGGLE THESE TWO LINES FOR THE TWO DIFFERENT FILES '''
# csv_file = "data/letter-recognition-clean.csv"
csv_file = "data/bank-additional-full-clean.csv"
X, y = u.load_csv(csv_file, has_headers=True)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

if csv_file == "data/bank-additional-full-clean.csv":
    ###### TUNING HYPER PARAMS ######
    cv = ShuffleSplit(n_splits=10, test_size=0.2)
    name = "K-Nearest Neighbor (Euclidean) - Bank"

    clf = neighbors.KNeighborsClassifier()
    krange = np.arange(1,50)
    u.make_val_curve(clf, name, X, y, param_range=krange, ylim=(.80, 1.0),
                        name_mod='euclidean-bank', param_name="n_neighbors",
                        param_string="K-Neighbors", cv=cv, show=False)


    clf = neighbors.KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    cv = ShuffleSplit(n_splits=10, test_size=0.2)
    u.plot_learning_curve(clf, name, X, y, cv=cv, n_jobs=None, show=False,
        name_mod='euclidean-bank', ylim=(0,1.1), train_sizes=np.linspace(.1, 1.0, 10))

    test_size = 0.20
    print("Benchmark Default KNN Test (Euclidean)")
    clf = neighbors.KNeighborsClassifier()
    u.test_single_classifier(clf, X, y, test_size=test_size)

    print ("Tuned KNN Test - Single Iteration (Euclidean)")
    clf = neighbors.KNeighborsClassifier(n_neighbors=40, metric='euclidean')
    u.test_single_classifier(clf, X, y, test_size=test_size)

    name = "K-Nearest Neighbor (chebyshev) - Bank"

    clf = neighbors.KNeighborsClassifier(metric='chebyshev')
    u.make_val_curve(clf, name, X, y, param_range=krange, ylim=None,
                        name_mod='chebyshev-bank', param_name="n_neighbors",
                        param_string="K-Neighbors", cv=cv, show=False)


    clf = neighbors.KNeighborsClassifier(n_neighbors=40, metric='chebyshev')
    u.plot_learning_curve(clf, name, X, y, cv=cv, n_jobs=None, show=False,
        name_mod='chebyshev-bank', ylim=(.5,1.0), train_sizes=np.linspace(.1, 1.0, 5))

    test_size = 0.20
    print("Benchmark Default KNN Test (chebyshev)")
    clf = neighbors.KNeighborsClassifier(metric='chebyshev')
    u.test_single_classifier(clf, X, y, test_size=test_size)

    print ("Tuned Tree Test - Single Iteration (chebyshev)")
    clf = neighbors.KNeighborsClassifier(metric='chebyshev', n_neighbors=40)
    u.test_single_classifier(clf, X, y, test_size=test_size)

elif csv_file == 'data/letter-recognition-clean.csv':
    cv = ShuffleSplit(n_splits=100, test_size=0.2)
    name = "K-Nearest Neighbor (Euclidean) - Letter"
    ###### TUNING HYPER PARAMS ######

    clf = neighbors.KNeighborsClassifier()
    krange = np.arange(1,11)
    u.make_val_curve(clf, name, X, y, param_range=krange, ylim=None,
                        name_mod='euclidean-letter', param_name="n_neighbors",
                        param_string="K-Neighbors", cv=cv, show=False)


    clf = neighbors.KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    u.plot_learning_curve(clf, name, X, y, cv=cv, show=False,
        name_mod='euclidean-letter', ylim=(0, 1.1),
        train_sizes=np.linspace(.1, 1.0, 10))

    test_size = 0.20
    print("Benchmark Default KNN Test (Euclidean)")
    clf = neighbors.KNeighborsClassifier()
    u.test_single_classifier(clf, X, y, test_size=test_size, cm=False)

    print ("Tuned KNN Test - Single Iteration (Euclidean)")
    clf = neighbors.KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    u.test_single_classifier(clf, X, y, test_size=test_size, cm=False)

    name = "K-Nearest Neighbor (chebyshev) - Letter"

    clf = neighbors.KNeighborsClassifier(metric='chebyshev')
    u.make_val_curve(clf, name, X, y, param_range=krange, ylim=None,
                        name_mod='chebyshev-letter', param_name="n_neighbors",
                        param_string="K-Neighbors", cv=cv, show=False)


    clf = neighbors.KNeighborsClassifier(metric='chebyshev', n_neighbors=1)
    u.plot_learning_curve(clf, name, X, y, cv=cv, show=False,
                                    name_mod='minkowski-letter', ylim=(0,1.1),
                                    train_sizes=np.linspace(.1, 1.0, 5))

    test_size = 0.20
    print("Benchmark Default KNN Test (chebyshev)")
    clf = neighbors.KNeighborsClassifier()
    u.test_single_classifier(clf, X, y, test_size=test_size, cm=False)

    print ("Tuned Tree Test - Single Iteration (chebyshev)")
    clf = neighbors.KNeighborsClassifier(metric='chebyshev', n_neighbors=1)
    u.test_single_classifier(clf, X, y, test_size=test_size, cm=False)

print ('Total Run Time in Seconds: ' + str(time.time() - start))
