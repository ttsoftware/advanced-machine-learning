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
            val = args[0]
            for i, x in enumerate(val):
                if not type(x) == DataPoint:
                    val[i] = DataPoint(x)
            super(DataSet, self).__init__(val)

        self.dimensions = len(np.array(self.unpack_params()).T)

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

        spike_range_start = 10
        spike_range_end = 20

        spike_size = spike_range_end - spike_range_start

        mean = np.mean(data_transposed, axis=tuple(range(1, data_transposed.ndim)))
        var = np.var(data_transposed, axis=tuple(range(1, data_transposed.ndim)))
        # cov = np.cov(data_transposed)

        # covariance matrix with smaller variance
        # divisor = np.array([0.1 for i in range(len(cov))])
        # cov_small = np.divide(cov, divisor)

        # sample from our gaussian
        # samples = np.random.multivariate_normal(mean, cov_small, spike_size)

        print  mean[0] + var[0] * np.sin((np.pi / spike_size) * 10)
        print data[0][0]

        for t in range(spike_range_start, spike_range_end):
            d = np.sin((np.pi / spike_size) * (t - spike_range_start))
            for position in range(len(data[t])):
                data[t][position] = mean[position] + var[position] * d / 6

        noise_dataset = DataSet(data.tolist())

        return noise_dataset, range(spike_range_start, spike_range_end)

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
