import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier # SGD Classifier: train instances one by one(online learning)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.base import clone, BaseEstimator

SAMPLE_IDX = 36000
TOTAL_SIZE = 0
TRAIN_SIZE = 60000

def show_img(sample):
    # show sample digit image w/ pyplot
    sample_resize = sample.reshape(28, 28)

    plt.imshow(sample_resize, cmap = matplotlib.cm.binary, interpolation= "nearest")
    # cmap: color map / interpolation: whether to interpolate btw pixels if the display resolution is not the same as the image resolution
    plt.axis("off")
    plt.show()

if __name__ == '__main__':
    # fetch mnist data
    mnist = fetch_openml('mnist_784')
    X, y = mnist['data'], mnist['target']
    TOTAL_SIZE = X.shape[0]
    print(X.shape, y.shape) # X: (70000, 784) y: (70000,)

    # shuffle and generate train, test set
    shuffle_idx = np.random.permutation(TOTAL_SIZE)
    X, y = X[shuffle_idx], y[shuffle_idx]
    X_train, y_train, X_test, y_test = X[:TRAIN_SIZE], y[:TRAIN_SIZE], X[TRAIN_SIZE:], y[TRAIN_SIZE:]

    # show sample digit image w/ pyplot
    X_sample = X[SAMPLE_IDX]
    y_sample = y[SAMPLE_IDX]
    show_img(X_sample)

    # binary classifier - 5 for not
    y_train_5 = (y_train == '5')
    y_test_5 = (y_test == '5')

    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(X_train, y_train_5)
    print("answer:", y_sample)
    print("prediction:", sgd_clf.predict([X_sample]))

    # measurement of binary classifier
    skfolds = StratifiedKFold(n_splits=3)

    # measurement 1 : stratifiedKFold
    for train_idx, check_idx in skfolds.split(X_train, y_train_5):
        clone_clf = clone(sgd_clf)
        X_train_folds = X_train[train_idx]
        y_train_fold = y_train_5[train_idx]
        X_check_folds = X_train[check_idx]
        y_check_fold = y_train_5[check_idx]

        clone_clf.fit(X_train_folds, y_train_fold)
        y_pred = clone_clf.predict(X_check_folds)
        n_correct = sum(y_pred == y_check_fold)
        print(n_correct/ len(y_pred)) # 0.9683 0.9635 0.9717

    # measurement 2 : cross_val_score
    print(cross_val_score(sgd_clf, X_train, y_train_5, cv = 3, scoring = "accuracy")) # [0.9683 0.9635 0.9717]

    # measurement 3 : compare with dumb classifier
    class Never5Classifier(BaseEstimator):
        def fit(self, X, y=None): pass
        def predict(self, X): return np.zeros((len(X), 1), dtype = bool)

    never_5_clf = Never5Classifier()
    print(cross_val_score(never_5_clf, X_train, y_train_5, cv =3, scoring="accuracy")) #[0.90765 0.9122  0.90985]








