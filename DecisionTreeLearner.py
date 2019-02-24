from sklearn import tree
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import util as u

print ("Running Decision Tree script...")

''' TOGGLE THESE TWO LINES FOR THE TWO DIFFERENT FILES '''
# csv_file = "data/letter-recognition-clean.csv"
csv_file = "data/bank-additional-full-clean.csv"
X, y = u.load_csv(csv_file, has_headers=True, shuffle=True)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

if csv_file == "data/bank-additional-full-clean.csv":

    cv = ShuffleSplit(n_splits=100, test_size=0.2)
    name = "Decision Tree - Bank"

    ###### TUNING HYPER PARAMS ######
    clf = tree.DecisionTreeClassifier()
    drange = np.arange(1,10)
    u.make_val_curve(clf, name, X, y, param_range=drange, ylim=None,
            param_name="max_depth", param_string="Max Depth", cv=cv, show=False)
    clf = tree.DecisionTreeClassifier()
    lrange = np.arange(2,50)
    u.make_val_curve(clf, name, X, y, param_range=lrange, ylim=None,
                    param_name="max_leaf_nodes", param_string="Max Leaf Nodes",
                                                            cv=cv, show=False)
    ###### Generating Learning Curve ######
    clf = tree.DecisionTreeClassifier(max_depth=5, max_leaf_nodes=13)
    cv = ShuffleSplit(n_splits=100, test_size=0.2)
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a valida. set
    u.plot_learning_curve(clf, name, X, y, cv=cv, n_jobs=None, show=False,
                           ylim=(0,1.1), train_sizes=np.linspace(.1, 1.0, 10))

    ###### Testing tuned clf ######
    test_size = 0.20
    print("Benchmark Default Tree Test")
    clf = tree.DecisionTreeClassifier()
    u.test_single_classifier(clf, X, y, test_size=test_size)

    print ("Tuned Tree Test - Single Iteration")
    clf = tree.DecisionTreeClassifier(max_depth = 5)
    u.test_single_classifier(clf, X, y, test_size=test_size, cm=True)

elif csv_file == 'data/letter-recognition-clean.csv':

    ###### TUNING HYPER PARAMS ######
    cv = ShuffleSplit(n_splits=50, test_size=0.2)
    name = "Decision Tree - Letter"
    clf = tree.DecisionTreeClassifier()
    drange = np.arange(1,50)
    u.make_val_curve(clf, name, X, y, param_range=drange, ylim=None,
            param_name="max_depth", param_string="Max Depth", cv=cv, show=False)
    clf = tree.DecisionTreeClassifier()
    lrange = np.arange(2,100)
    u.make_val_curve(clf, name, X, y, param_range=lrange, ylim=None,
                    param_name="max_leaf_nodes", param_string="Max Leaf Nodes",
                                                            cv=cv, show=False)

    clf = tree.DecisionTreeClassifier()
    u.plot_learning_curve(clf, name, X, y, cv=cv, show=False, ylim=(0, 1.1),
                            train_sizes=np.linspace(.1, 1.0, 10))

    ###### Testing tuned clf ######
    test_size = 0.20
    print("Benchmark Default Tree Test (same as tuned)")
    clf = tree.DecisionTreeClassifier()
    u.test_single_classifier(clf, X, y, test_size=test_size, cm=False)
