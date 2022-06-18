# from turtle import distance
import pandas
import matplotlib.pyplot as plt
import numpy as np

names = ['sex', 'length', 'diameter', 'height', 'whole', 'shucked', 'viscera', 'shell', 'rings']
data = pandas.read_csv('abalone.data', names=names)
# print(data)

# data["rings"].hist()
# plt.show()

# clear dataset
data = data.drop('sex', axis=1)

# search correlation
correlation = data.corr().drop('rings', axis=0)

X = data.drop('rings', axis=1).values
Y = data['rings'].values

# print(X)
# print(Y)

new_data_point = np.array(correlation['rings'])
distances = np.linalg.norm(X - new_data_point, axis=1)

k = 3
nearest_neighbor_ids = distances.argsort()[:k]
nearest_neighbor_rings = Y[nearest_neighbor_ids]
print(nearest_neighbor_rings)

# x_train_ids = np.random.randint(Y.shaper[0], size=round(len(Y)*0.8))
# x_test_ids = np.random.randint(Y.shaper[0], size=round(len(Y)*0.2))
# print(len(x_train_ids), len(x_test_ids))

# def splitTrainTest(data, percent):
#     trainData = []
#     trainDataLen = round(len(data) * (percent / 100))
#     for row in data:
#         trainData

# def knn(u, XY, w):
#     labels = []
#     XY_sort = sort(X, key=euclidean_distance(X(i)))
#     for i in XY_sort[

