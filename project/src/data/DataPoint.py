import numpy as np

from decimal import Decimal


class DataPoint(object):

    def __init__(self, params, target=None):

        if type(params) != list:  # assume it is ndarray vector [[]]
            params = map(lambda x: float(x[0]), params.tolist())

        if target is not None:
            self.target = float(target)
        else:
            self.target = None

        # we cannot convert complex to float
        self.params = map(lambda x: float(x), params)

    def get_vector(self):
        return np.array(map(lambda x: [x], self.params[:]))

    def __ne__(self, other):
        return self.params != other.params
