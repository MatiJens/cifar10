from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV

from utils.load_file import load_file

train_raw_file_path = "data/cifar10/train"
test_raw_file_path = "data/cifar10/test"

processed_file_path = "data/cifar10/processed"

x_train_filename = "x_train.npy"
y_train_filename = "y_train.npy"
x_test_filename = "x_test.npy"
y_test_filename = "y_test.npy"

x_train, y_train = load_file(train_raw_file_path, processed_file_path, x_train_filename, y_train_filename)
x_test, y_test = load_file(test_raw_file_path, processed_file_path, x_test_filename, y_test_filename)

print("Data loading finished\n")

"""
#------------------- LINEAR KERNEL -------------------
param_grid = {
    'C' : [0.01, 0.1, 1.0, 10.0],
    'penalty' : ['l1', 'l2']
}

base_clf = svm.LinearSVC(random_state=7, dual=False, max_iter=5000, verbose=0)

grid_search = GridSearchCV(base_clf, param_grid, cv=3, verbose=2, n_jobs=2)
grid_search.fit(x_train, y_train)

best_clf = grid_search.best_estimator_

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best accuracy: {grid_search.best_score_}")

test_predict = best_clf.predict(x_test)

print("Prediction finished")

"""
"""
#------------------- RBF KERNEL -------------------

param_grid = {
    'C' : [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    'gamma' : [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
}

base_clf = svm.SVC(kernel='rbf', random_state=7, cache_size=500)

grid_search = GridSearchCV(base_clf, param_grid, cv=3, verbose=2, n_jobs=2)
grid_search.fit(x_train, y_train)

best_clf = grid_search.best_estimator_

print(f"Best parameters: {grid_search.best_params_}\n")

test_predict = best_clf.predict(x_test)
"""

clf = svm.SVC(C=1000.0, gamma=0.0001, kernel='rbf', random_state=7, cache_size=500, verbose=2)
clf.fit(x_train, y_train)

print("Model learing finished\n")

test_predict = clf.predict(x_test)

accuracy = accuracy_score(y_test, test_predict)
f1 = f1_score(y_test, test_predict, average='weighted')

cifar10_class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']

classification_report = classification_report(y_test, test_predict, target_names=cifar10_class_names)

print(f"Accuracy: {accuracy:.4f}\n")
print(f"F1 score (weighted): {f1:.4f}\n")
print("Classification report:\n")
print(classification_report)