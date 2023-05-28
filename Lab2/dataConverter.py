import pandas as pd
import numpy as np


class DataConverter:
    names = ['family', 'product-type', 'steel', 'carbon', 'hardness', 'temper_rolling', 'condition',
             'formability', 'strength', 'non-ageing', 'surface-finish', 'surface-quality', 'enamelability',
             'bc', 'bf', 'bt', 'bw/me', 'bl', 'm', 'chrom', 'phos', 'cbond', 'marvi', 'exptl', 'ferro',
             'corr', 'blue/bright/varn/clean', 'lustre', 'jurofm', 's', 'p', 'shape', 'thick', 'width',
             'len', 'oil', 'bore', 'packing', 'classes']

    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.data_train = pd.read_csv('data/anneal.data', names=self.names)
        self.data_test = pd.read_csv('data/anneal.test', names=self.names)

    def convert(self):
        self.X_train = self.data_train.drop('classes', axis=1)
        self.y_train = self.data_train.classes

        self.X_test = self.data_test.drop('classes', axis=1)
        self.y_test = self.data_test.classes

        # clean missing value
        for column in self.X_train.columns.values:
            if list(self.X_train[column]).count('?') == 798:
                self.X_train.drop(column, axis=1)
                self.X_test.drop(column, axis=1)
            else:
                self.X_train[column] = self.X_train[column].replace('?', np.nan)
                self.X_test[column] = self.X_test[column].replace('?', np.nan)

        return self.X_train, self.X_test, self.y_train, self.y_test
