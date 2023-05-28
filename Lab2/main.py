import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from dataConverter import DataConverter

converter = DataConverter()
X_train, X_test, y_train, y_test = converter.convert()

clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy on train set: {:3f}".format(clf.score(X_train, y_train)))
print("Accuracy on test set: {:3f}".format(clf.score(X_test, y_test)))
