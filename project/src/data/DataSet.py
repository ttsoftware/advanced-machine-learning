import numpy as np

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
            for i, x in enumerate(args[0]):
                if not type(x) == DataPoint:
                    raise TypeError('Objects must be of type DataPoint')
            super(DataSet, self).__init__(args[0])

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

    def principal_component(self, k=2, component_variance=0.8, centroids=None):
        """
        Returns a new dataset reduced to k principal components (dimensions)
        :type component_variance: float The threshold for component variance
        :param centroids: Expects centroids to be of type dataset
        :rtype : DataSet
        :param k:
        """
        assert k < self.dimensions

        covariance = np.cov(np.array(self.unpack_params()).T)

        # eigenvectors and eigenvalues for the from the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(covariance)

        sorted_eig = map(lambda (i, x): (x, eigenvectors[i]), enumerate(eigenvalues))
        sorted_eig = sorted(sorted_eig, key=lambda e: e[0], reverse=True)

        if not k:
            eigenvaluesum = sum(eigenvalues)
            eigenvaluethreshold = eigenvaluesum * component_variance

            cumsum_sorted_eig = 0
            sorted_eig_threshold_index = 0
            for i in range(len(sorted_eig)):
                if cumsum_sorted_eig < eigenvaluethreshold:
                    cumsum_sorted_eig += sorted_eig[i][0]
                else:
                    sorted_eig_threshold_index = i
                    break

            W = np.array([sorted_eig[i][1] for i in range(sorted_eig_threshold_index)])
        else:
            # we choose the largest eigenvalues
            W = np.array([sorted_eig[i][1] for i in range(k)])

        return DataSet(
            map(
                lambda x: DataPoint(np.dot(W, x.params).tolist(), x.target),
                self if not centroids else centroids
            )
        )

    def add_artifacts(self, k=None):
        columns = np.array(self.unpack_params()).T

        column_variances = map(lambda c: np.var(c), columns)
        column_means = map(lambda c: np.mean(c), columns)

        artifacts = []

        for i in range(k):
            z = np.random.randn()
            artifact = []
            for c in range(len(columns)):
                artifact += [(column_means[c] + z) * column_variances[c]]
            artifacts += [DataPoint(artifact)]

        return artifacts

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