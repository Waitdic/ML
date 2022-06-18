from cmath import sqrt
import pandas
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

names = ['sex', 'length', 'diameter', 'height', 'whole', 'shucked', 'viscera', 'shell', 'rings']
data = pandas.read_csv('C:\\git\\mlvs\\Lab1\\dara\\abalone.data', names=names).drop('sex', axis=1)

X = data.drop('rings', axis=1).values
Y = data['rings'].values

# Раделяю выборку на тренировочный и тестовый набор
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)



