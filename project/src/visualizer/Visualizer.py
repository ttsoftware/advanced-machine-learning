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

        :param window_size:
        :param original_dataset:
        :param reconstructed_dataset:
        :param name:
        :return:
        """

        original_windows = ExperimentorService.windows(original_dataset.clone(), window_size)
        reconstructed_windows = ExperimentorService.windows(reconstructed_dataset.clone(), window_size)
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

    @staticmethod
    def visualize_cross_validation_bars(original_dataset, artifact_dataset, thresholds, window_sizes, name='figure_cross_validation_bars'):
        """

        :param artifact_dataset:
        :param thresholds:
        :param original_dataset:
        :param window_sizes:
        :param name:
        :return:
        """

        # Do cross validation
        mse = {thresholds[0]: [], thresholds[1]: [], thresholds[2]: []}

        for threshold in thresholds:
            for window_size in window_sizes:
                original_windows = ExperimentorService.windows(original_dataset.clone(), window_size)
                artifact_windows = ExperimentorService.windows(artifact_dataset.clone(), window_size)

                current_mse = []
                for idx, original_window in enumerate(original_windows):
                    reconstructed_window, rejected = ExperimentorService.pca_reconstruction(artifact_windows[idx], window_size, threshold)

                    current_mse += [ExperimentorService.mse(original_window, reconstructed_window)]

                mse[threshold] += [np.mean(current_mse)]

        fig, ax = plt.subplots()

        indexs = np.arange(len(mse[thresholds[0]]))
        width = 0.20

        ax.bar(indexs, mse[thresholds[0]], width, label='Max eigenvalue threshold', color='c', alpha=0.8)
        ax.bar(indexs + width, mse[thresholds[1]], width, label='Average eigenvalue threshold', color='b', alpha=0.8)
        ax.bar(indexs + width*2, mse[thresholds[2]], width, label='Average of max eigenvalue threshold', color='m', alpha=0.8)

        ax.set_xticks(indexs + width*1.5)
        ax.set_xticklabels([str(window_size) for window_size in window_sizes])

        ax.set_title('mse cross validation')
        ax.set_ylabel('Mean squared error')
        ax.set_xlabel('Window size')

        plt.legend(loc='upper right')
        plt.savefig(name)

    @staticmethod
    def visualize_cross_validation_curves(original_dataset, artifact_dataset, thresholds, window_sizes, name='figure_cross_validation_curves'):
        """

        :param artifact_dataset:
        :param thresholds:
        :param original_dataset:
        :param window_sizes:
        :param name:
        :return:
        """

        # Do cross validation
        mse = {thresholds[0]: [], thresholds[1]: [], thresholds[2]: []}

        for threshold in thresholds:
            for window_size in window_sizes:
                original_windows = ExperimentorService.windows(original_dataset.clone(), window_size)
                artifact_windows = ExperimentorService.windows(artifact_dataset.clone(), window_size)

                current_mse = []
                for idx, original_window in enumerate(original_windows):
                    reconstructed_window, rejected = ExperimentorService.pca_reconstruction(artifact_windows[idx],
                                                                                            window_size, threshold)

                    current_mse += [ExperimentorService.mse(original_window, reconstructed_window)]

                mse[threshold] += [np.mean(current_mse)]

        fig, ax = plt.subplots()

        ax.plot(mse[thresholds[0]], label='Max eigenvalue threshold', color='c')
        ax.plot(mse[thresholds[1]], label='Average eigenvalue threshold', color='b')
        ax.plot(mse[thresholds[2]], label='Average of max eigenvalue threshold', color='m')

        ax.set_xticks(range(len(window_sizes)))
        ax.set_xticklabels([str(window_size) for window_size in window_sizes])

        ax.set_title('mse cross validation')
        ax.set_ylabel('Mean squared error')
        ax.set_xlabel('Window size')

        plt.legend(loc='upper right')
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

    @staticmethod
    def visualize_all(original, noisy, reconstructed, name='figure_comparison', components=14):
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
            axarr[index].plot(np.array(original.unpack_params()).T[i], color='y')
            axarr[index].plot(np.array(noisy.unpack_params()).T[i], color='r')
            axarr[index].plot(np.array(reconstructed.unpack_params()).T[i], color='b')

        plt.savefig(name, papertype='a0', pad_inches=0, bbox_inches=0, frameon=False)

    @staticmethod
    def visualize_timeLine(original, noisy, reconstructed, name='timeline'):
        """
        Visualizes the original dataset alongside the dataset with added artifacts
        and the reconstructed dataset.

        :param name: the name of the image.
        :param components: How many components should be realized, starting from component 0
        :return: None
        """


        f = plt.figure(figsize=(10,4))
        f.subplots_adjust(bottom=0.2)
        ax = f.add_subplot(111)
        ax.plot(np.array(original.unpack_params()).T[13], label='Original signal')
        ax.set_xlabel('Time (1/60 seconds)')
        ax.set_ylabel('Amplitude')
        legend = ax.legend(loc='upper right')
        plt.savefig('Original_Signal')

        f = plt.figure(figsize=(10,4))
        f.subplots_adjust(bottom=0.2)
        ax = f.add_subplot(111)
        ax.plot(np.array(original.unpack_params()).T[13], label='Original signal')
        ax.plot(np.array(noisy.unpack_params()).T[13], label='Noisy signal')
        ax.set_xlabel('Time (1/60 seconds)')
        ax.set_ylabel('Amplitude')
        legend = ax.legend(loc='upper right')
        plt.savefig('With_artifacts')

        f = plt.figure(figsize=(10,4))
        f.subplots_adjust(bottom=0.2)
        ax = f.add_subplot(111)
        ax.plot(np.array(original.unpack_params()).T[13], label='Original signal')
        ax.plot(np.array(reconstructed.unpack_params()).T[13], label='Reconstructed signal')
        ax.set_xlabel('Time (1/60 seconds)')
        ax.set_ylabel('Amplitude')
        legend = ax.legend(loc='upper right')
        plt.savefig('After_PCA')


    @staticmethod
    def visualize_mse_on_same(original_dataset, reconstructed_dataset_max,reconstructed_dataset_avg,reconstructed_dataset_avg_max, window_size, name='figure_mse'):
        """

        :param window_size:
        :param original_dataset:
        :param reconstructed_dataset:
        :param name:
        :return:
        """

        f = plt.figure(figsize=(10,4))
        f.subplots_adjust(bottom=0.2)
        ax = f.add_subplot(111)

        original_windows = ExperimentorService.windows(original_dataset.clone(), window_size)
        reconstructed_windows = ExperimentorService.windows(reconstructed_dataset_max.clone(), window_size)
        mse_windows = []

        for idx, original_window in enumerate(original_windows):
            mse = ExperimentorService.mse(original_window, reconstructed_windows[idx])
            mse_windows.append(mse)

        ax.plot(mse_windows, label='Maximum Threshold')

        reconstructed_windows = ExperimentorService.windows(reconstructed_dataset_avg.clone(), window_size)
        mse_windows = []

        for idx, original_window in enumerate(original_windows):
            mse = ExperimentorService.mse(original_window, reconstructed_windows[idx])
            mse_windows.append(mse)

        ax.plot(mse_windows, label='Average Threshold')

        reconstructed_windows = ExperimentorService.windows(reconstructed_dataset_avg_max.clone(), window_size)
        mse_windows = []

        for idx, original_window in enumerate(original_windows):
            mse = ExperimentorService.mse(original_window, reconstructed_windows[idx])
            mse_windows.append(mse)

        ax.plot(mse_windows, label='Maximum average Threshold')


        ax.set_xlabel('Window #)')
        ax.set_ylabel('Mean Squared Error')
        legend = ax.legend(loc='upper right')

        plt.savefig('Mean_Squared_Comparison')
