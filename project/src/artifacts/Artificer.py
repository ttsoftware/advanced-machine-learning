import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import NearestNeighbors

import src.artifacts.pca.PcaProjector as PCA

from src.data.DataSet import DataSet
from src.data.Normalizer import Normalizer


class Artificer:
    def __init__(self, dataset, add_artifacts=False):
        self.spike_range_end = 35
        self.spike_range_start = 5

        self.original_dataset = dataset
        if add_artifacts:
            self.has_artifacts = True
            self.noise_dataset = self.add_artifacts()
        else:
            self.has_artifacts = False
            self.noise_dataset = dataset.clone()

        self.normalizer = Normalizer(dataset.clone())
        self.normalized_noise_dataset = self.normalizer.subtract_means(self.noise_dataset.clone())

        self.reconstructed_dataset = None

    def add_artifacts(self, k=None):
        """
        Adds k noisy artifacts to self.
        :param k:
        :return:
        """
        data = np.array(self.original_dataset.unpack_params())
        data_transposed = data.T

        mean = np.mean(data_transposed, axis=tuple(range(1, data_transposed.ndim)))
        var = np.var(data_transposed, axis=tuple(range(1, data_transposed.ndim)))

        self.sine_artifact(self.spike_range_start, self.spike_range_end, data, mean, var)

        noise_dataset = DataSet(data.tolist())

        return noise_dataset

    def sine_artifact(self, spike_range_start, spike_range_end, data, mean, var):

        self.factors = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 3, 4, 40, 100, 300, 800, 400, 50, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # np.linspace(1, 80, len(data[0]))

        for t in range(spike_range_start, spike_range_end):
            d = np.sin((np.pi / (spike_range_end - spike_range_start)) * (t - spike_range_start))

            for position in range(len(data[t])):
                data[t][position] += d * self.factors[position]

    def put_artifacts(self):
        self.noise_dataset = self.add_artifacts()
        self.normalized_noise_dataset = self.normalizer.subtract_means(self.noise_dataset.clone())

    def pca_reconstruction(self, threshold=None):
        """
        Does PCA projection on the normalized noise dataset in order to reconstruct the original
        dataset with artifacts removed

        TODO: proper threshold that is not based on percentage

        :param threshold: The threshold where the PCA projector will reject principal components
        :return:
        """
        reconstructed_dataset, avg_eigenvalue, max_eigenvalue, rejected = PCA.project(self.normalized_noise_dataset, threshold)
        self.reconstructed_dataset = self.normalizer.add_means(reconstructed_dataset)

        return avg_eigenvalue, max_eigenvalue, rejected

    def visualize(self, name='figure_artifacts', components=14):
        """
        Visualizes the original dataset alongside the dataset with added artifacts
        and the reconstructed dataset.

        :param name: the name of the image.
        :param components: How many components should be realized, starting from component 0
        :return: None
        """
        f, axarr = plt.subplots(components, 1)
        axarr[0].set_title('Corrected EEG')
        axarr[0].ticklabel_format(useOffset=False)

        for index, i in enumerate(range(components)):
            axarr[index].plot(np.array(self.original_dataset.unpack_params()).T[i], color='y')
            axarr[index].plot(np.array(self.noise_dataset.unpack_params()).T[i], color='r')
            axarr[index].plot(np.array(self.reconstructed_dataset.unpack_params()).T[i], color='b')

        plt.savefig(name, papertype='a0', pad_inches=0, bbox_inches=0, frameon=False)

    def mse(self):
        new_data = np.array(self.original_dataset.unpack_params())
        old_data = np.array(self.reconstructed_dataset.unpack_params())
        sum_all_dataset = 0
        sum_with_artifacts = 0
        sum_without_artifacts = 0
        nb_datapoints_with_artifacts = 0
        nb_datapoints_without_artifacts = 0

        for i in range(len(new_data)):

            for j in range(len(new_data[i])):

                sum_all_dataset += np.power(new_data[i][j] - old_data[i][j], 2)

                if self.has_artifacts and self.spike_range_start <= i <= self.spike_range_end:

                    if self.factors[j] > 0:
                        sum_with_artifacts += np.power(new_data[i][j] - old_data[i][j], 2)
                        nb_datapoints_with_artifacts += 1
                    else:
                        sum_without_artifacts += np.power(new_data[i][j] - old_data[i][j], 2)
                        nb_datapoints_without_artifacts += 1
                else:
                    sum_without_artifacts += np.power(new_data[i][j] - old_data[i][j], 2)
                    nb_datapoints_without_artifacts += 1

        if nb_datapoints_with_artifacts == 0:
            mse_with_artifacts = 0
        else:
            mse_with_artifacts = sum_with_artifacts/nb_datapoints_with_artifacts

        return sum_all_dataset/(len(new_data) * len(new_data)), sum_without_artifacts/nb_datapoints_without_artifacts, 0

    def knn_threshold(self):

        data = np.array(self.noise_dataset.unpack_params()).T
        covariance = np.cov(data)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        print eigenvectors

        knn = NearestNeighbors(n_neighbors=3, algorithm='kd_tree')
        model = knn.fit(eigenvectors)

        distance_graph = model.kneighbors_graph(mode='distance').toarray()
        # remove the sparse values
        # we now have i -> [distance to K nearest neighbors]
        distance_matrix = np.array(map(lambda x: filter(lambda y: y != 0, x), distance_graph))

        print distance_matrix
        mean_distance = np.mean(distance_matrix)
        # mean_distance = np.mean(distance_matrix, axis=tuple(range(1, data.ndim)))
        print mean_distance

        # find all points, whose nearest-neighbor-distance-mean is more than the mean distance away
        print np.array(filter(lambda x: np.mean(x) > mean_distance, distance_matrix))

        # plt.scatter(X, c='k', label='data')
        # plt.plot(y_, c='g', label='prediction')
        # plt.axis('tight')
        # plt.legend()
        # plt.show()

        return 0
