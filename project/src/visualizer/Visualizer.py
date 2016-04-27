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
        :param window_size:
        :param name:
        :return:
        """

        dataset = self.dataset.clone()

        mse_windows = []

        threshold = 0
        for idx in range(calibration_length):
            current_window = DataSet(dataset[idx * window_size:(idx + 1) * window_size])

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

        for idx in range(calibration_length, len(dataset) // window_size):
            current_window = DataSet(dataset[idx * window_size:(idx + 1) * window_size])

            artificer = Artificer(current_window, add_artifacts)
            artificer.pca_reconstruction(threshold)
            mse = artificer.mse()
            mse_windows.append(mse)

        f = plt.figure()
        ax = f.add_subplot(111)
        ax.plot(mse_windows)
        ax.set_xlabel('window #')
        ax.set_ylabel('Mean Squared Error')
        plt.savefig(name + '_' + threshold_type)

    def visualize_cross_validation(self, calibration_length, window_sizes, add_artifacts=True, color='r', name='figure_cross_validation'):
        """

        :param color:
        :param width:
        :param calibration_length:
        :param window_sizes:
        :param add_artifacts:
        :param name:
        :return:
        """

        dataset = self.dataset.clone()

        # Calibration
        threshold_max = 0
        threshold_avg = 0
        threshold_avg_max = 0
        for idx in range(calibration_length):
            current_window = DataSet(dataset[idx * 40:(idx + 1) * 40])

            artificer = Artificer(current_window, add_artifacts=False)
            avg_eigenvalue, max_eigenvalue, rejected = artificer.pca_reconstruction()

            threshold_max = max(threshold_max, max_eigenvalue)
            threshold_avg = max(threshold_avg, avg_eigenvalue)
            threshold_avg_max += max_eigenvalue

        threshold_avg_max = np.mean(threshold_avg_max)

        thresholds = [threshold_max, threshold_avg, threshold_avg_max]

        # Make dataset with artifacts
        new_dataset = DataSet()
        for idx in range(calibration_length, len(dataset) // 40):
            current_window = DataSet(dataset[idx * 40:(idx + 1) * 40])

            artificer = Artificer(current_window, add_artifacts)
            new_dataset += artificer.get_noise_dataset()

        # Do cross validation
        mse = [0] * (len(window_sizes) * 3)
        parameter_combo = 0

        for threshold in thresholds:
            for window_size in window_sizes:
                current_mse = 0
                windows = range(calibration_length, len(dataset) // window_size)
                for window_idx in windows:
                    current_window = DataSet(dataset[window_idx * window_size:(window_idx + 1) * window_size])

                    artificer = Artificer(current_window, add_artifacts=False)
                    artificer.pca_reconstruction(threshold)
                    current_mse += artificer.mse()
                mse[parameter_combo] = current_mse / len(windows)
                parameter_combo += 1

        f = plt.figure()
        ax = f.add_subplot(111)
        rects = ax.bar(np.arange(len(mse)), mse, color=color)

        ax.set_ylabel('Mean squared error')
        ax.set_title('mse cross validation')
        ax.set_xticks(np.arange(len(mse)))
        labels = ['max_' + str(window_size) for window_size in window_sizes] + ['avg_' + str(window_size) for window_size in window_sizes] + ['avg-max_' + str(window_size) for window_size in window_sizes]
        ax.set_xticklabels(labels)
        plt.xticks(rotation=70)

        plt.savefig(name)

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