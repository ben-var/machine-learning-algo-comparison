import util
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ShuffleSplit

csv_file = "data/letter-recognition-clean.csv"
dataset_name = 'Letter'
# csv_file = "data/bank-additional-full-clean.csv"
# dataset_name = 'Bank'
X, y = util.load_csv(csv_file, shuffle=True, has_headers=True)


scaler = MinMaxScaler()
X = scaler.fit_transform(X)

cv = ShuffleSplit(n_splits=20, test_size=0.2)

get_roc = True
check_time = True
check_error = True

clfs, clf_names = util.get_tuned_clfs(csv_file)

if get_roc:
    print ("Generating ROC Curve for " + csv_file)
    clfs, clf_names = util.get_tuned_clfs(csv_file)
    file_name = 'graphs/' + dataset_name + '_roc.png'
    util.generate_ROC(X, y, clfs, clf_names, dataset_name, save_as=file_name, test_size=0.20)
if check_time:
    clfs, clf_names = util.get_tuned_clfs(csv_file)
    print ("Time dict for " + csv_file)
    print (util.get_train_test_time(X, y, clfs, clf_names, test_size=0.20))
if check_error:
    print ("Error dict for " + csv_file)
    clfs, clf_names = util.get_tuned_clfs(csv_file)
    print (util.get_error_rates(X, y, clfs, clf_names, cv=cv, test_size=0.20))
