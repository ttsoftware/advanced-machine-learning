import numpy as np
import matplotlib.pyplot as plt
import src.artifacts.pca.PcaProjector as PCA

from src.Data.DataSet import DataSet
from src.Data.Normalizer import Normalizer


class Artificer:
    def __init__(self, dataset, add_artifacts=False):
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
        self.spike_range_start = 5
        self.spike_range_end = 35
        self.sine_artifact(self.spike_range_start, self.spike_range_end, data, mean, var)

        noise_dataset = DataSet(data.tolist())

        return noise_dataset

    def sine_artifact(self, spike_range_start, spike_range_end, data, mean, var):

        self.factors = [0, 0, 0, 0, 1, 1, 1, 1, 3, 4, 40, 100, 300, 800] # np.linspace(1, 80, len(data[0]))

        for t in range(spike_range_start, spike_range_end):
            d = np.sin((np.pi / (spike_range_end - spike_range_start)) * (t - spike_range_start))

            for position in range(len(data[t])):
                data[t][position] += d * self.factors[position]

    def pca_reconstruction(self, threshold=None):
        """
        Does PCA projection on the normalized noise dataset in order to reconstruct the original
        dataset with artifacts removed

        TODO: proper threshold that is not based on percentage

        :param threshold: The threshold where the PCA projector will reject principal components
        :return:
        """
        reconstructed_dataset, max_eigenvalue, rejected = PCA.project(self.normalized_noise_dataset, threshold)
        self.reconstructed_dataset = self.normalizer.add_means(reconstructed_dataset)

        return max_eigenvalue, rejected

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

    def mse(self):
        new_data = np.array(self.original_dataset.unpack_params())
        old_data = np.array(self.reconstructed_dataset.unpack_params())
        sum_all_dataset = 0
        sum_with_artifacts =0
        sum_without_artifacts =0
        nb_datapoints_with_artifacts = 0
        nb_datapoints_without_artifacts = 0
        for i in range(len(new_data)):
           for j in range(len(new_data[i])):
                sum_all_dataset += np.power(new_data[i][j] - old_data[i][j],2)
                if(self.has_artifacts and i>=self.spike_range_start and i<=self.spike_range_end):
                    if(self.factors[j]>0):
                        sum_with_artifacts += np.power(new_data[i][j] - old_data[i][j],2)
                        nb_datapoints_with_artifacts +=1
                    else:
                        sum_without_artifacts += np.power(new_data[i][j] - old_data[i][j],2)
                        nb_datapoints_without_artifacts +=1
                else:
                    sum_without_artifacts += np.power(new_data[i][j] - old_data[i][j],2)
                    nb_datapoints_without_artifacts +=1

        return sum_all_dataset, sum_with_artifacts, sum_without_artifacts, nb_datapoints_with_artifacts, nb_datapoints_without_artifacts
