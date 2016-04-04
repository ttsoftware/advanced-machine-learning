import numpy as np

from random import randrange
from scipy.optimize import curve_fit

from DataPoint import DataPoint


class DataSet(list):
    def __init__(self, *args, **kwargs):
        """
        :param args: List of DataPoints
        :param kwargs:
        """

        self.class_sets = {}
        self.dimensions = 0  # number of dimensions in each datapoint

        if len(args) > 0:
            val = args[0]
            for i, x in enumerate(val):
                if not type(x) == DataPoint:
                    val[i] = DataPoint(x)
            super(DataSet, self).__init__(val)

    def unpack_params(self):
        """
        Get the parameters from our dataset
        """
        values = []
        for i, data_point in enumerate(self):
            values += [data_point.params[:]]

        return values

    def unpack_numpy_array(self):
        """
        Get the parameters from our dataset as numpy arrays
        """
        values = []
        for i, data_point in enumerate(self):
            values += [data_point.get_vector()]
        return values

    def unpack_targets(self):
        """
        Get the targets for our dataset
        """
        return map(lambda x: x.target, self)

    def project_pca(self, k=2, component_variance=0.8, centroids=None):
        """
        Returns a new dataset reduced to k principal components (dimensions)
        :type component_variance: float The threshold for principal component variance
        :param centroids: Expects centroids to be of type dataset
        :rtype : DataSet
        :param k:
        """
        assert k < self.dimensions

        data_transposed = np.array(self.unpack_params()).T

        covariance = np.cov(data_transposed)

        # eigenvectors and eigenvalues for the from the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        sorted_eig = map(lambda (i, x): (x, eigenvectors[i]), enumerate(eigenvalues))
        sorted_eig = sorted(sorted_eig, key=lambda e: e[0], reverse=False)

        if not k:
            eigenvaluesum = sum(eigenvalues)
            eigenvaluethreshold = eigenvaluesum * component_variance

            cumsum_sorted_eig = 0
            sorted_eig_threshold_index = 0
            for i in range(len(sorted_eig)):
                if (cumsum_sorted_eig + sorted_eig[i][0]) < eigenvaluethreshold:
                    cumsum_sorted_eig += sorted_eig[i][0]
                else:
                    sorted_eig_threshold_index = i
                    break

            W = np.array([sorted_eig[i][1] for i in range(sorted_eig_threshold_index)])
            W = np.append(W, [np.zeros(len(data_transposed))], axis=0)
        else:
            # we choose the smallest eigenvalues
            W = np.array([sorted_eig[i][1] for i in range(k)])

        return DataSet(
            map(
                lambda x: DataPoint(np.dot(W, x.params).tolist()),
                self if not centroids else centroids
            )
        )

    def add_artifacts(self, k=None):
        """
        Adds k noisy artifacts to self.
        :param k:
        :return:
        """
        data = np.array(self.unpack_params())
        data_transposed = data.T

        # random spike interval
        # spike_range_start = randrange(0, len(rows))
        # spike_range_end = randrange(spike_range_start, (spike_range_start + len(rows)))

        spike_range_start = 30
        spike_range_end = 50
        spike_size = spike_range_end - spike_range_start

        mean = np.mean(data_transposed, axis=tuple(range(1, data_transposed.ndim)))
        cov = np.cov(data_transposed)

        divisor = np.array([0.01 for i in range(len(cov))])

        cov_big = np.divide(cov, divisor)

        samples = np.random.multivariate_normal(mean, cov_big, spike_size)

        data[spike_range_start:spike_range_end] += samples

        noise_dataset = DataSet(data.tolist())

        return noise_dataset, range(spike_range_start, spike_range_end)

    def project_W(self, W):
        Winv = np.linalg.pinv(W)
        return DataSet(
            map(
                lambda row: DataPoint(np.dot(Winv, row).tolist()),
                self.unpack_params()
            )
        )

    def sort(self, cmp=None, key=None, reverse=False):
        super(DataSet, self).sort(cmp=cmp, key=lambda x: x.params[0], reverse=reverse)

    def __iadd__(self, other):
        if not type(other[0]) == DataPoint:
            raise TypeError('Objects must be of type DataPoint')

        if self.dimensions is 0:
            self.dimensions = len(other[0].params)

        return super(DataSet, self).__iadd__(other)

    def clone(self):
        """
        Fuck Value-by-reference bullshit
        :return:
        """
        return DataSet(self[:])
