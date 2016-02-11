from __future__ import division
import numpy as np
import Regression as Regression


class MAPRegression(Regression.Regression):

    def __init__(self, alpha, beta, dataset):
        super(MAPRegression, self).__init__(dataset)

        S_N = np.linalg.inv(alpha * np.identity(self.d_mat.shape[1]) + beta * np.dot(self.d_mat.T, self.d_mat))

        M_N = beta * np.dot(np.dot(S_N, self.d_mat.T), self.t_vec)
        self.w = M_N

    def predict(self, input_vector):
        """
        Predicts a target based on the N-dimensional input vector x
        :param input_vector:
        :return:
        """
        return np.dot(self.w.T, input_vector)[0]