import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from classifier import Classifier


def loo():
    k_scores = {}

    for k in range(20, 31, 2):
        knn = Classifier(k)
        knn.fit(X_train, y_train)
        pred = knn.predict(X_test)
        score = np.sum(pred == y_test) / len(y_test)
        k_scores.update({k: score})
        print('k: ', k, 'score: ', score)

    return k_scores


def sclearn(k_range):
    knn = KNeighborsClassifier(n_neighbors=k_range)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred)
    print('sc_learn_k: ', k_range, 'score: ', score)
    return


names = ['sex', 'length', 'diameter', 'height', 'whole', 'shucked', 'viscera', 'shell', 'rings']
import_data = pd.read_csv('abalone.data', names=names)

data = import_data.drop('sex', axis=1)

X = data.drop('rings', axis=1).values
y = data.rings.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.28, random_state=46)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

clf = Classifier(21)
clf.fit(X_train, y_train)
clf.predict(X_test)

precision = clf.precision(y_test)
recall = clf.recall(y_test)

k_metrics = loo()

plt.plot(k_metrics.keys(), k_metrics.values())
plt.xlabel('K')
plt.ylabel('score')

sclearn(max(k_metrics, key=k_metrics.get))
