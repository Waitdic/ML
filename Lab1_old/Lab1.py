from turtle import distance
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


# def splitTrainTest(data, percent):
#     trainData = []
#     trainDataLen = round(len(data) * (percent / 100))
#     for row in data:
#         trainData
