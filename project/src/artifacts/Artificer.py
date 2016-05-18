import numpy as np
from sklearn.neighbors import NearestNeighbors

import src.artifacts.pca.PcaProjector as PCA

from src.data.DataSet import DataSet
from src.data.Normalizer import Normalizer


class Artificer:
    def __init__(self, dataset_window):
        # self.factors = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 3, 4, 40, 100, 300, 800, 400, 50, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # np.linspace(1, 80, len(data[0]))
        self.factors = [0, 0, 0, 0, 1, 1, 1, 1, 3, 4, 40, 80, 100, 800]
        #self.factors = [0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 150]
        # self.factors = map(lambda x: x*2, self.factors)
        self.window = dataset_window
        self.spike_range_start = None
        self.spike_range_end = None
        self.normalizer = None

    def add_artifacts(self, spike_size=30):
        """
        Adds k noisy artifacts to self.
        :param k:
        :return:
        """
        spike_offset = (len(self.window) - spike_size) // 2

        self.spike_range_start = spike_offset
        self.spike_range_end = spike_size + spike_offset

        data = np.array(self.window.unpack_params())
        data_transposed = data.T

        mean = np.mean(data_transposed, axis=tuple(range(1, data_transposed.ndim)))
        var = np.var(data_transposed, axis=tuple(range(1, data_transposed.ndim)))

        self.sine_artifact(self.spike_range_start, self.spike_range_end, data, mean, var)

        noise_dataset = DataSet(data.tolist())

        return noise_dataset

    def sine_artifact(self, spike_range_start, spike_range_end, data, mean, var):
        sinus = []

        for t in range(spike_range_start, spike_range_end):
            d = np.sin((2 * np.pi / (spike_range_end - spike_range_start)) * (t - spike_range_start))

            for position in range(len(data[t])):
                data[t][position] =  data[t][position] + d * self.factors[position]

    def pca_reconstruction(self, dataset=None, threshold=None):
        """
        Does PCA projection on the normalized noise dataset in order to reconstruct the original
        dataset with artifacts removed

        TODO: proper threshold that is not based on percentage

        :param dataset:
        :param threshold: The threshold where the PCA projector will reject principal components
        :return:
        """
        if dataset is None:
            dataset = self.window.clone()

        self.normalizer = Normalizer(dataset)
        normalized_noised_window = self.normalizer.subtract_means(dataset)

        reconstructed_dataset, avg_eigenvalue, max_eigenvalue, rejected = PCA.project(normalized_noised_window, threshold)
        reconstructed_window = self.normalizer.add_means(reconstructed_dataset)

        return reconstructed_window, avg_eigenvalue, max_eigenvalue, rejected

    def mse(self):
        original_data = np.array(self.window.unpack_params())
        reconstructed_data = np.array(self.reconstructed_window.unpack_params())
        sum_window = 0
        components_with_artifacts = 0

        for i in range(len(original_data)):

            for j in range(len(original_data[i])):

                sum_window += np.power(original_data[i][j] - reconstructed_data[i][j], 2)

                if self.factors[j] > 0:
                    components_with_artifacts += 1

        if components_with_artifacts == 0:
            # no artifacts in this window
            pass

        # MSE for this window
        return np.mean(sum_window)

    def get_noised_window(self):
        return self.noised_window.clone()
