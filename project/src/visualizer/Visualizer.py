import random

import matplotlib.pyplot as plt
import numpy as np

from src.artifacts.Artificer import Artificer
from src.data.DataSet import DataSet


class Visualizer:
    def __init__(self, dataset):
        self.dataset = dataset

    def visualize_mse(self, calibration_length, threshold_type, window_size, add_artifacts=True, name='figure_mse'):
        """

        :param add_artifacts:
        :param threshold_type:
        :param calibration_length: The number of windows used for calibration
        :param threshold:
        :param window_size:
        :param name:
        :return:
        """

        nb_windows = len(self.dataset) // window_size
        mse_windows = []

        threshold = 0
        for idx in range(calibration_length):
            current_window = DataSet(self.dataset[idx * window_size:(idx + 1) * window_size])

            if idx < calibration_length:
                artificer = Artificer(current_window, add_artifacts=False)
                avg_eigenvalue, max_eigenvalue, rejected = artificer.pca_reconstruction()

                if threshold_type == 'max':
                    threshold = max(threshold, max_eigenvalue)
                elif threshold_type == 'avg':
                    threshold = max(threshold, avg_eigenvalue)
                elif threshold_type == 'avg_max':
                    threshold += max_eigenvalue

        if threshold_type == 'avg_max':
            threshold = np.mean(threshold)

        for idx in range(calibration_length, nb_windows):
            current_window = DataSet(self.dataset[idx * window_size:(idx + 1) * window_size])

            artificer = Artificer(current_window, add_artifacts)
            artificer.pca_reconstruction(threshold)
            mse = artificer.mse()
            mse_windows.append(mse)

        print mse_windows

        plt.plot(mse_windows)
        plt.xlabel('window #')
        plt.ylabel('Mean Squared Error')
        plt.savefig(name + '_' + threshold_type)

    def visualize_data(self, name='figure_artifacts_data', components=14):
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