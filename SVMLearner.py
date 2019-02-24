from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import util as u
import time

start = time.time()

print ("Running SVMLearner script...")

''' TOGGLE THESE TWO LINES FOR THE TWO DIFFERENT FILES '''
csv_file = 'data/letter-recognition-clean.csv'
# csv_file = "data/bank-additional-full-clean.csv"
X, y = u.load_csv(csv_file, shuffle=True, has_headers=True)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

if csv_file == "data/bank-additional-full-clean.csv":

    ###### TUNING HYPER PARAMS ######
    cv = ShuffleSplit(n_splits=5, test_size=0.2)
    name = "SVC (Linear) - Bank"
    clf = SVC(kernel='linear', gamma='auto', max_iter=100000)
    crange = np.arange(1,10) * 15
    u.make_val_curve(clf, name, X, y, param_range=crange,
        name_mod="lin-bank", param_name="C", param_string="C (Penalty)",
                                            ylim=None, cv=cv, show=False)

    clf = LinearSVC(max_iter=20000)
    u.plot_learning_curve(clf, name, X, y, cv=cv, show=False,
                        ylim=(0.8,1.0), train_sizes=np.linspace(.1, 1.0, 5))

    test_size = 0.20
    print("Benchmark Default LinearSVC Test - Bank")
    clf = LinearSVC()
    u.test_single_classifier(clf, X, y, test_size=test_size)

    test_size = 0.20
    print("Tuned LinearSVC Test - Bank")
    clf = LinearSVC()
    u.test_single_classifier(clf, X, y, test_size=test_size)

    cv = ShuffleSplit(n_splits=5, test_size=0.2)
    name = "SVC (Rbf) - Bank"
    clf = SVC(kernel='rbf', gamma='auto', max_iter=100000)
    u.make_val_curve(clf, name, X, y, param_range=crange, ylim=None,
        name_mod='rbf-bank', param_name="C", param_string="C (Penalty)",
                                                    cv=cv, show=False)

    clf = SVC(kernel='rbf', gamma='auto', max_iter=500000, C=15)
    u.plot_learning_curve(clf, name, X, y, cv=cv, n_jobs=None, show=False,
                            ylim=(0,1.1), train_sizes=np.linspace(.1, 1.0, 5))

    test_size = 0.20
    print("Benchmark Default SVC Test (rbf) - Bank")
    clf = SVC(kernel='rbf', gamma='auto')
    u.test_single_classifier(clf, X, y, test_size=test_size)

    print("Tuned SVC Test (rbf) - Bank")
    clf = SVC(kernel='rbf', gamma='auto')
    u.test_single_classifier(clf, X, y, test_size=test_size)

elif csv_file == 'data/letter-recognition-clean.csv':

    ###### TUNING HYPER PARAMS ######
    cv = ShuffleSplit(n_splits=5, test_size=0.2)
    name = "SVC (Linear) - Letter"
    clf = LinearSVC(max_iter=10000)
    crange = [1000, 2000, 3000, 4000, 5000]
    u.make_val_curve(clf, name, X, y, param_range=crange, ylim=None,
        name_mod="lin", param_name="C", param_string="C (Penalty)",
                                                            cv=cv, show=False)

    clf = LinearSVC()
    u.plot_learning_curve(clf, name, X, y, cv=cv, n_jobs=None, show=False,
                                        train_sizes=np.linspace(.1, 1.0, 5))

    test_size = 0.20
    print("Benchmark Default SVC Test - Bank")
    clf = LinearSVC()
    u.test_single_classifier(clf, X, y, test_size=test_size, cm=False)

    name = "SVC (Rbf) - Letter"
    clf = SVC(kernel='rbf', gamma='auto', max_iter=100000)
    u.make_val_curve(clf, name, X, y, param_range=crange, ylim=None,
        name_mod='rbf', param_name="C", param_string="C (Penalty)",
                                                    cv=cv, show=False)

    clf = SVC(kernel='rbf', gamma='auto', C=5000, max_iter=200000)
    u.plot_learning_curve(clf, name, X, y, cv=cv, n_jobs=None, show=False,
                            ylim=(0,1.1), train_sizes=np.linspace(.1, 1.0, 5))

    test_size = 0.20
    print("Benchmark Default SVC Test (rbf)")
    clf = SVC(kernel='rbf')
    u.test_single_classifier(clf, X, y, test_size=test_size, cm=False)

    print("Tuned SVC Test (rbf)")
    clf = SVC(kernel='rbf', gamma='auto', C=5000, max_iter=100000)
    u.test_single_classifier(clf, X, y, test_size=test_size, cm=False)

print ('Time to run: ' + str(time.time() - start) + ' seconds.')
