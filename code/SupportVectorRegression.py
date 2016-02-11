from __future__ import division
from sklearn import svm
import numpy as np


class SupportVectorRegression(object):

    def __init__(self, dataset, gamma=0.2, C=10, kernel='rbf'):
        self.dataset = dataset.clone()

        self.features = self.dataset.unpack_params()
        self.targets = self.dataset.unpack_targets()

        self.clf = svm.SVR(kernel=kernel, gamma=gamma, C=C)
        self.clf.fit(self.features, self.targets)

    def predict(self, dataset):
        """
        Predicts the label for all features in the inputset
        """
        return self.clf.predict(dataset.unpack_params())

    def mean_square(self, inputset):
        targets = inputset.unpack_targets()
        predictions = self.predict(inputset)

        N = len(inputset)

        guess_sum = 0
        for i in range(N):
            guess_sum += (targets[i] - predictions[i]) ** 2

        return (1 / N) * guess_sum

    def root_mean_square(self, inputset):
        targets = inputset.unpack_targets()
        predictions = self.predict(inputset)

        N = len(inputset)

        guess_sum = 0
        for i in range(N):
            guess_sum += (targets[i] - predictions[i]) ** 2

        return np.sqrt((1 / N) * guess_sum)

    @staticmethod
    def cross_validator(dataset, gammas, Cs, s_fold=5, kernel='rbf'):

        s_partitions = int(len(dataset) / s_fold)

        test_partitions = []
        test_targets = []
        train_partitions = []
        train_targets = []

        for k in xrange(s_fold):
            start_current = k * s_partitions
            end_current = (k + 1) * s_partitions

            test_partitions += [map(lambda x: x.params, dataset[start_current:end_current])]
            test_targets += [map(lambda x: x.target, dataset[start_current:end_current])]
            train_partitions += [map(lambda x: x.params, (dataset[:start_current] + dataset[end_current:]))]
            train_targets += [map(lambda x: x.target, (dataset[:start_current] + dataset[end_current:]))]

        best_set = (-1, -1, 2)
        for C in Cs:
            for gamma in gammas:

                rms = []
                for h in range(len(train_partitions)):

                    clf = svm.SVR(kernel=kernel, gamma=gamma, C=C)
                    clf.fit(train_partitions[h], train_targets[h])

                    predictions = clf.predict(test_partitions[h])

                    guess_sum = 0
                    for k in range(len(predictions)):
                        guess_sum += (test_targets[h][k] - predictions[k]) ** 2

                    rms += [np.sqrt((1 / len(predictions)) * guess_sum)]

                avg_rms = sum(rms) / len(rms)
                if avg_rms < best_set[2]:
                    best_set = (gamma, C, avg_rms)

        # Return the amount of neighbors that yields the best loss
        return best_set