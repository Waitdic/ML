from knnModel import KnnModel
import numpy as np


def euclidean_dist(x, x_train):
    return np.sqrt(np.sum(x - x_train) ** 2)


class Classifier(KnnModel):

    def predict(self, x_test):
        self.predicted_list = np.array([self.predict_y(x) for x in x_test])
        return self.predicted_list

    def predict_y(self, x):
        distances = [euclidean_dist(x, X_train) for X_train in self.X_train]
        indexes = np.argsort(distances)[:self.k]
        nearest_labels = [self.y_train[i] for i in indexes]
        max_dist = max([distances[i] for i in indexes])

        if max_dist == 0: return nearest_labels[0]

        classes = {nearest_labels[0]: 0}
        for i in range(self.k):
            if classes.keys().__contains__(nearest_labels[i]):
                classes[nearest_labels[i]] += 1
            else:
                classes.update({nearest_labels[i]: 1})

        return max(classes, key=classes.get)

    def precision(self, y_test):
        y_unique = np.unique(y_test)
        presicion_result = {}

        for y in y_unique:
            tp = 0
            fp = 0

            for i in range(len(self.predicted_list)):
                if self.predicted_list[i] == y & y_test[i] == y:
                    tp += 1
                elif self.predicted_list[i] == y & y_test[i] != y:
                    fp += 1

            res = 0
            if fp + tp != 0:
                res = tp / (tp + fp)

            print('class: ', y, 'presicion: ', res)
            presicion_result.update({y: res})

        return presicion_result

    def recall(self, y_test):
        y_unique = np.unique(y_test)
        recall_result = {}

        for y in y_unique:
            tp = 0
            fn = 0

            for i in range(len(self.predicted_list)):
                if self.predicted_list[i] == y & y_test[i] == y:
                    tp += 1
                elif self.predicted_list[i] != y & y_test[i] == y:
                    fn += 1

            res = 0
            if fn + tp != 0:
                res = tp/(tp + fn)

            print('class: ', y, 'recall: ', res)
            recall_result.update({y: res})

        return recall_result

