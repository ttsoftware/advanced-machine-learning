import numpy as np
import matplotlib.pyplot as plt
import src.artifacts.pca.PcaProjector as PCA

from src.data.DataSet import DataSet
from src.data.Normalizer import Normalizer


class Artificer:
    def __init__(self, dataset, add_artifacts=False):
        self.original_dataset = dataset
        self.noise_dataset = self.add_artifacts()

        self.normalizer = Normalizer(dataset)
        self.normalized_noise_dataset = self.normalizer.subtract_means(self.noise_dataset if add_artifacts else dataset)

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


        mean = np.mean(data_transposed, axis=tuple(range(1, data_transposed.ndim)))
        var = np.var(data_transposed, axis=tuple(range(1, data_transposed.ndim)))

        self.sine_artifact(10, 14, data, mean, var)

        noise_dataset = DataSet(data.tolist())

        return noise_dataset

    def sine_artifact(self, spike_range_start, spike_range_end, data, mean, var):

        factors = np.linspace(1, 80, len(data[0]))

        for t in range(spike_range_start, spike_range_end):
            d = np.sin((np.pi / (spike_range_end - spike_range_start)) * (t - spike_range_start))

            for position in range(len(data[t])):
                data[t][position] += d * factors[position]

    def pca_reconstruction(self, threshold=None):
        """
        Does PCA projection on the normalized noise dataset in order to reconstruct the original
        dataset with artifacts removed

        TODO: proper threshold that is not based on percentage

        :param threshold: The threshold where the PCA projector will reject principal components
        :return:
        """
        reconstructed_dataset, max_eigenvalue = PCA.project(self.normalized_noise_dataset, threshold)
        self.reconstructed_dataset = self.normalizer.add_means(reconstructed_dataset)
        return max_eigenvalue

    def visualize(self, components=14):
        """
        Visualizes the original dataset alongside the dataset with added artifacts
        and the reconstructed dataset.

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

        plt.savefig("figure_artifact", papertype='a0', pad_inches=0, bbox_inches=0, frameon=False)
