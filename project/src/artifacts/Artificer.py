import numpy as np
from sklearn.neighbors import NearestNeighbors

import src.artifacts.pca.PcaProjector as PCA

from src.data.DataSet import DataSet
from src.data.Normalizer import Normalizer


class Artificer:
    def __init__(self, dataset_window, spike_size=30, add_artifacts=False):
        # self.factors = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 3, 4, 40, 100, 300, 800, 400, 50, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # np.linspace(1, 80, len(data[0]))
        self.factors = [0, 0, 0, 0, 1, 1, 1, 1, 3, 4, 40, 100, 300, 800]

        spike_offset = (len(dataset_window) - spike_size) // 2

        self.spike_range_start = spike_offset
        self.spike_range_end = spike_size + spike_offset

        self.original_window = dataset_window
        if add_artifacts:
            self.has_artifacts = True
            self.noised_window = self.add_artifacts()
        else:
            self.has_artifacts = False
            self.noised_window = dataset_window.clone()

        self.normalizer = Normalizer(dataset_window.clone())
        self.normalized_noised_window = self.normalizer.subtract_means(self.noised_window.clone())

        self.reconstructed_window = None

    def add_artifacts(self, k=None):
        """
        Adds k noisy artifacts to self.
        :param k:
        :return:
        """
        data = np.array(self.original_window.unpack_params())
        data_transposed = data.T

        mean = np.mean(data_transposed, axis=tuple(range(1, data_transposed.ndim)))
        var = np.var(data_transposed, axis=tuple(range(1, data_transposed.ndim)))

        self.sine_artifact(self.spike_range_start, self.spike_range_end, data, mean, var)

        noise_dataset = DataSet(data.tolist())

        return noise_dataset

    def sine_artifact(self, spike_range_start, spike_range_end, data, mean, var):
        for t in range(spike_range_start, spike_range_end):
            d = np.sin((2 * np.pi / (spike_range_end - spike_range_start)) * (t - spike_range_start))

            for position in range(len(data[t])):
                data[t][position] += d * self.factors[position]

    def put_artifacts(self):
        self.noised_window = self.add_artifacts()
        self.normalized_noised_window = self.normalizer.subtract_means(self.noised_window.clone())

    def pca_reconstruction(self, threshold=None):
        """
        Does PCA projection on the normalized noise dataset in order to reconstruct the original
        dataset with artifacts removed

        TODO: proper threshold that is not based on percentage

        :param threshold: The threshold where the PCA projector will reject principal components
        :return:
        """
        reconstructed_dataset, avg_eigenvalue, max_eigenvalue, rejected = PCA.project(self.normalized_noised_window, threshold)
        self.reconstructed_window = self.normalizer.add_means(reconstructed_dataset)

        return avg_eigenvalue, max_eigenvalue, rejected

    def mse(self):
        original_data = np.array(self.original_window.unpack_params())
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

    def get_reconstructed_window(self):
        return self.reconstructed_window.clone()
