from __future__ import division
import numpy as np


class Regression(object):
    def __init__(self, dataset):
        """
        prepends 1's onto each row in the design matrix, and runs regressions,
        to find w.

        :param d_mat (Design matric):
        :param t_vec (Target vector:
        """

        dataset = dataset.clone()
        d_mat = map(lambda x: x.params, dataset)
        t_vec = map(lambda x: [x.target], dataset)

        self.d_mat = np.array(map(lambda x: [1] + x, d_mat))
        self.t_vec = np.array(t_vec)

    def predict(self, x):
        raise Exception('Not yet implemented')

    def mean_square(self, dataset=None):
        """
        Find the dimension_means-square error for the given regression
        :param: dataset (Optional dataset to compare against)
        :return:
        """
        if dataset is not None:
            N = len(dataset)
        else:
            N = len(self.d_mat)

        guess_sum = 0
        for i in range(N):
            if dataset is not None:
                guess_sum += (dataset[i].target - self.predict([1] + dataset[i].params)) ** 2
            else:
                guess_sum += (self.t_vec[i][0] - self.predict(self.d_mat[i])) ** 2

        return (1 / N) * guess_sum

    def root_mean_square(self, dataset=None):
        """
        Find the root-dimension_means-square error for the given regression
        :param: dataset (Optional dataset to compare against)
        :return:
        """
        if dataset is not None:
            N = len(dataset)
        else:
            N = len(self.d_mat)

        guess_sum = 0
        for i in range(N):
            if dataset is not None:
                guess_sum += (dataset[i].target - self.predict([1] + dataset[i].params)) ** 2
            else:
                guess_sum += (self.t_vec[i][0] - self.predict(self.d_mat[i])) ** 2

        return np.sqrt((1 / N) * guess_sum)