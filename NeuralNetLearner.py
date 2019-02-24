from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import util as u
import time

start = time.time()

print ("Running Neural Net script...")

''' TOGGLE THESE TWO LINES FOR THE TWO DIFFERENT FILES '''
# csv_file = "data/bank-additional-full-clean.csv"
csv_file = "data/letter-recognition-clean.csv"
X, y = u.load_csv(csv_file, shuffle=True, has_headers=True)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

if csv_file == "data/bank-additional-full-clean.csv":

    cv = ShuffleSplit(n_splits=5, test_size=0.2)
    name = "Neural Net (relu) - Bank"
    clf = MLPClassifier(hidden_layer_sizes=(15,2))
    irange = np.arange(1,11) * 100
    u.make_val_curve(clf, name, X, y, param_range=irange, ylim=None,
        param_name="max_iter", param_string="Max Iterations", cv=cv, show=False)

    clf = MLPClassifier(max_iter=5000)
    hrange = []
    cust_range = []
    for i in range(20,31):
        hrange.append((i,15,2))
        cust_range.append(i)
    u.make_val_curve(clf, name, X, y, param_range=hrange,
        ylim=None, custom_param_range=cust_range,
        param_name="hidden_layer_sizes", name_mod='4layer-bank',
        param_string="First Hidden Layer Size", cv=cv, show=False)

    clf = MLPClassifier(activation='logistic', max_iter = 5000, hidden_layer_sizes=(15,2))
    u.plot_learning_curve(clf, name, X, y, cv=cv, n_jobs=None, show=False,
                            ylim=(0,1.1),train_sizes=np.linspace(.1, 1.0, 5))

    test_size = 0.20
    print("Benchmark Default NN Test - Bank")
    clf = MLPClassifier(activation='logistic')
    u.test_single_classifier(clf, X, y, test_size=test_size)

    test_size = 0.20
    print("Tuned NN Test - Bank")
    clf = MLPClassifier(activation='logistic', max_iter = 5000, hidden_layer_sizes=(15,2))
    u.test_single_classifier(clf, X, y, test_size=test_size)

elif csv_file == 'data/letter-recognition-clean.csv':

    cv = ShuffleSplit(n_splits=3, test_size=0.2)
    name = "Neural Net (relu) - Letter"
    clf = MLPClassifier(100,50,26)
    irange = np.arange(1,20) * 100
    u.make_val_curve(clf, name, X, y, param_range=irange, ylim=None,
        param_name="max_iter", param_string="Max Iterations", cv=cv, show=False,
        name_mod='letter')

    clf = MLPClassifier(max_iter=5000)
    hrange = []
    for i in range(1,6):
        hrange.append((i*20,80,50, 26))
    u.make_val_curve(clf, name, X, y, param_range=hrange, ylim=None,
        custom_param_range= np.arange(1,6) * 20,
        param_name="hidden_layer_sizes", name_mod='-letter(4layer)',
        param_string="First Layer Size (4 layer net)", cv=cv, show=False)


    clf = MLPClassifier(max_iter=5000, hidden_layer_sizes=(80, 50, 26))
    u.plot_learning_curve(clf, name, X, y, cv=cv, n_jobs=None, show=False, ylim=(0,1.1),
                                            name_mod='letter', train_sizes=np.linspace(.1, 1.0, 5))

    name = "Neural Net (logistic) - Letter"
    ''' Testing the logistic activation function '''
    clf = MLPClassifier(activation='logistic', max_iter=5000, hidden_layer_sizes=(80, 50, 26))
    u.plot_learning_curve(clf, name, X, y, cv=cv, n_jobs=None, show=False, ylim=(0,1.1),
                                            name_mod='letter', train_sizes=np.linspace(.1, 1.0, 5))

    ''' Simple benchmark test followed by a tuned learner test '''
    test_size = 0.20
    print("Benchmark Default NN Test - Letter")
    clf = MLPClassifier()
    u.test_single_classifier(clf, X, y, test_size=test_size, cm=False)

    test_size = 0.20
    print("Tuned NN Test - Letter")
    clf = MLPClassifier(max_iter=5000, hidden_layer_sizes=(80, 50, 26))
    u.test_single_classifier(clf, X, y, test_size=test_size, cm=False)


print ('Total Run Time in Seconds: ' + str(time.time() - start))
