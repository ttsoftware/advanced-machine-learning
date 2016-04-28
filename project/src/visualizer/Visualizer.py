import random

import matplotlib.pyplot as plt
import numpy as np

from src.artifacts.Artificer import Artificer
from src.data.DataSet import DataSet
from src.experimentor.ExperimentorService import ExperimentorService


class Visualizer:

    def __init__(self):
        pass

    @staticmethod
    def visualize_mse(original_dataset, reconstructed_dataset, window_size, name='figure_mse'):
        """

        :param add_artifacts:
        :param threshold_type:
        :param calibration_length: The number of windows used for calibration
        :param window_size:
        :param name:
        :return:
        """

        original_windows = ExperimentorService.windows(original_dataset.clone())
        reconstructed_windows = ExperimentorService.windows(reconstructed_dataset.clone())
        mse_windows = []

        for idx, original_window in enumerate(original_windows):
            mse = ExperimentorService.mse(original_window, reconstructed_windows[idx])
            mse_windows.append(mse)

        f = plt.figure()
        ax = f.add_subplot(111)
        ax.plot(mse_windows)
        ax.set_xlabel('window #')
        ax.set_ylabel('Mean Squared Error')
        plt.savefig(name)

    def visualize_cross_validation(self, calibration_length, window_sizes, random=True, color='r', name='figure_cross_validation'):
        """

        :param color:
        :param width:
        :param calibration_length:
        :param window_sizes:
        :param add_artifacts:
        :param name:
        :return:
        """

        experimentor = ExperimentorService(self.dataset.clone())
        threshold_max, threshold_avg, threshold_avg_max = experimentor.calibrate(calibration_length)
        thresholds = [threshold_max, threshold_avg, threshold_avg_max]

        artificers = experimentor.artifactify(random)

        # Do cross validation
        mse = [0] * (len(window_sizes) * 3)
        parameter_combo = 0

        for threshold in thresholds:
            for window_size in window_sizes:
                current_mse = 0
                windows = range(calibration_length, len(artificer))
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