import numpy as np
import matplotlib.pyplot as plt
import src.artifacts.pca.PcaProjector as PCA

from src.data.DataSet import DataSet
from src.data.Normalizer import Normalizer


class Artificer:
    def __init__(self, dataset):
        self.original_dataset = dataset
        self.noise_dataset, self.spike_range = self.add_artifacts()

        self.normalizer = Normalizer(self.noise_dataset)
        self.normalized_noise_dataset = self.normalizer.subtract_means(self.noise_dataset)

        self.reconstructed_dataset = None

    def add_artifacts(self, k=None):
        """
        Adds k noisy artifacts to self.
        :param k:
        :return:
        """
        data = np.array(self.original_dataset.unpack_params())
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

        print mean[0] + var[0] * np.sin((np.pi / spike_size) * 10)
        print data[0][0]

        for t in range(spike_range_start, spike_range_end):
            d = np.sin((np.pi / spike_size) * (t - spike_range_start))
            for position in range(len(data[t])):
                data[t][position] = mean[position] + var[position] * d / 6

        noise_dataset = DataSet(data.tolist())

        return noise_dataset, range(spike_range_start, spike_range_end)

    def pca_reconstruction(self, threshold=1):
        """
        Does PCA projection on the normalized noise dataset in order to reconstruct the original
        dataset with artifacts removed

        TODO: proper threshold that is not based on percentage

        :param threshold: The threshold where the PCA projector will reject principal components
        :return:
        """
        reconstructed_dataset = PCA.project(self.normalized_noise_dataset, threshold)
        self.reconstructed_dataset = self.normalizer.add_means(reconstructed_dataset)

    def visualize(self, components=10):
        """
        Visualizes the original dataset alongside the dataset with added artifacts
        and the reconstructed dataset.

        :param components: How many components should be realized, starting from component 0
        :return: None
        """
        f, axarr = plt.subplots(components, 3)
        axarr[0, 0].set_title('Original EEG')
        axarr[0, 1].set_title('Noised EEG')
        axarr[0, 2].set_title('Corrected EEG')

        for index, i in enumerate(range(components)):
            axarr[index, 0].plot(np.array(self.original_dataset.unpack_params()).T[i])
            axarr[index, 1].plot(np.array(self.noise_dataset.unpack_params()).T[i])
            axarr[index, 2].plot(np.array(self.reconstructed_dataset.unpack_params()).T[i])

        # pca_dataset_columns = np.array(projection_dataset.unpack_params()).T

        # for idx, j in enumerate(pca_dataset_columns):
        #    axarr[idx, 3].plot(j)

        # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
        plt.setp([a.get_xticklabels() for a in axarr[1, :]], visible=False)
        plt.setp([a.get_yticklabels() for a in axarr[:, 2]], visible=True)
        plt.show()
