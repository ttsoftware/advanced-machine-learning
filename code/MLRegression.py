from __future__ import division
import numpy as np

import Regression as Regression


class MLRegression(Regression.Regression):

    def __init__(self, dataset):
        super(MLRegression, self).__init__(dataset)

        self.w = self.regression()

    def regression(self):
        """
        Returns the dot product of the pseudo-inverse of the design matrix,
        and the target vector.
        """
        return np.dot(np.linalg.pinv(self.d_mat), self.t_vec)

    def predict(self, input_vector):
        """
        Returns the dot product of the transposed w, and the N-dimensional design matrix x.
        """
        return sum(np.dot(self.w.T, input_vector))