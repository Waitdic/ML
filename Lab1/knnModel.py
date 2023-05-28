class KnnModel:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.predicted_list = []

    def fit(self, x, y):
        self.X_train = x
        self.y_train = y
