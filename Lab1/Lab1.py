import pandas
import matplotlib.pyplot as plt

names = ['sex', 'length', 'diameter', 'height', 'whole', 'shucked', 'viscera', 'shell', 'rings']
data = pandas.read_csv('abalone.data', names=names)
# print(data)

# data["rings"].hist()
# plt.show()

# clear dataset
data = data.drop('sex', axis=1)

# search correlation
correlation = data.corr()

X = data.drop('rings', axis=1).values
Y = data['rings'].values


print(X)
print(Y)

def splitTrainTest(data, percent):
    trainData = []
    trainDataLen = round(len(data) * (percent / 100))
    for row in data:
        trainData
