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
    def visualize_cross_validation_bars(training_set, test_set, artifact_dataset, window_sizes, name='figure_cross_validation_bars'):
        """

        :param training_set:
        :param artifact_dataset:
        :param test_set:
        :param window_sizes:
        :param name:
        :return:
        """

        # Do cross validation
        mse = {'max': [], 'avg': [], 'avg_max': []}

        for window_size in window_sizes:
            for i, threshold in enumerate(ExperimentorService.calibrate(training_set, window_size)):

                original_windows = ExperimentorService.windows(test_set.clone(), window_size)
                artifact_windows = ExperimentorService.windows(artifact_dataset.clone(), window_size)

                current_mse = []
                for idx, original_window in enumerate(original_windows):
                    reconstructed_window, rejected = ExperimentorService.pca_reconstruction(artifact_windows[idx], window_size, threshold)

                    current_mse += ExperimentorService.mse(original_window, reconstructed_window)

                if i == 0:
                    mse['max'] += [np.mean(current_mse)]
                elif i == 1:
                    mse['avg'] += [np.mean(current_mse)]
                else:
                    mse['avg_max'] += [np.mean(current_mse)]

        best_index_max = mse['max'].index(min(mse['max']))
        best_index_avg = mse['avg'].index(min(mse['avg']))
        best_index_avg_max = mse['avg_max'].index(min(mse['avg_max']))

        print 'Best window size for max threshold: ' + str(window_sizes[best_index_max])
        print 'Best window size for avg threshold: ' + str(window_sizes[best_index_avg])
        print 'Best window size for avg_max threshold: ' + str(window_sizes[best_index_avg_max])

        fig, ax = plt.subplots()

        indexs = np.arange(len(mse['max']))
        width = 0.20

        ax.bar(indexs, mse['max'], width, label='Max eigenvalue threshold', color='c', alpha=0.8)
        ax.bar(indexs + width, mse['avg'], width, label='Average eigenvalue threshold', color='b', alpha=0.8)
        ax.bar(indexs + width*2, mse['avg_max'], width, label='Average of max eigenvalue threshold', color='m', alpha=0.8)
        ax.set_ylim([0,1500])

        ax.set_xticks(indexs + width*1.5)
        ax.set_xticklabels([str(window_size) for window_size in window_sizes])
        plt.xticks(rotation=70)

        ax.set_title('mse cross validation')
        ax.set_ylabel('Mean squared error')
        ax.set_xlabel('Window size')

        plt.legend(loc='upper right')
        plt.savefig(name)

    @staticmethod
    def visualize_cross_validation_bars_percentage(training_set, test_set, artifact_dataset, window_sizes,
                                                   name='figure_cross_validation_bars_difference'):
        """

        :param artifact_dataset:
        :param thresholds:
        :param test_set:
        :param window_sizes:
        :param name:
        :return:
        """

        # Do cross validation
        differences = {'max': [], 'avg': [], 'avg_max': []}

        for window_size in window_sizes:
            for i, threshold in enumerate(ExperimentorService.calibrate(training_set, window_size)):
                original_windows = ExperimentorService.windows(test_set.clone(), window_size)
                artifact_windows = ExperimentorService.windows(artifact_dataset.clone(), window_size)

                current_difference = []
                for idx, original_window in enumerate(original_windows):
                    reconstructed_window, rejected = ExperimentorService.pca_reconstruction(artifact_windows[idx],
                                                                                            window_size, threshold)

                    current_difference += ExperimentorService.difference(original_window, reconstructed_window)

                if i == 0:
                    differences['max'] += [np.mean(current_difference)]
                elif i == 1:
                    differences['avg'] += [np.mean(current_difference)]
                else:
                    differences['avg_max'] += [np.mean(current_difference)]

                print 'threshold: ' + differences.keys()[i] + ' - window size: ' + str(window_size) + ' - difference: ' + str(np.mean(current_difference))

        fig, ax = plt.subplots()

        indexs = np.arange(len(differences['max']))
        width = 0.20

        ax.bar(indexs, differences['max'], width, label='Max eigenvalue threshold', color='c', alpha=0.8)
        ax.bar(indexs + width, differences['avg'], width, label='Average eigenvalue threshold', color='b', alpha=0.8)
        ax.bar(indexs + width * 2, differences['avg_max'], width, label='Average of max eigenvalue threshold', color='m',
               alpha=0.8)

        ax.set_xticks(indexs + width * 1.5)
        ax.set_xticklabels([str(window_size) for window_size in window_sizes])
        plt.xticks(rotation=70)

        ax.set_title('Difference cross validation')
        ax.set_ylabel('Difference %')
        ax.set_xlabel('Window size')

        plt.legend(loc='upper right')
        plt.savefig(name)

    @staticmethod
    def visualize_cross_validation_curves(training_set, test_set, artifact_dataset, window_sizes, name='figure_cross_validation_curves'):
        """

        :param training_set:
        :param artifact_dataset:
        :param test_set:
        :param window_sizes:
        :param name:
        :return:
        """

        # Do cross validation
        mse = {'max': [], 'avg': [], 'avg_max': []}

        for window_size in window_sizes:
            for i, threshold in enumerate(ExperimentorService.calibrate(training_set, window_size)):
                original_windows = ExperimentorService.windows(test_set.clone(), window_size)
                artifact_windows = ExperimentorService.windows(artifact_dataset.clone(), window_size)

                current_mse = []
                for idx, original_window in enumerate(original_windows):
                    reconstructed_window, rejected = ExperimentorService.pca_reconstruction(artifact_windows[idx],
                                                                                            window_size, threshold)

                    current_mse += ExperimentorService.mse(original_window, reconstructed_window)

                if i == 0:
                    mse['max'] += [np.mean(current_mse)]
                elif i == 1:
                    mse['avg'] += [np.mean(current_mse)]
                else:
                    mse['avg_max'] += [np.mean(current_mse)]

        fig, ax = plt.subplots()

        ax.plot(mse['max'], label='Max eigenvalue threshold', color='c')
        ax.plot(mse['avg'], label='Average eigenvalue threshold', color='b')
        ax.plot(mse['avg_max'], label='Average of max eigenvalue threshold', color='m')

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
    def visualize_timeLine(original, test, noisy, reconstructed, name='timeline'):
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
        plt.axvline(x=280, color='black')
        ax.set_xlabel('Time (1/128 seconds)')
        ax.set_ylabel('Amplitude')
        legend = ax.legend(loc='upper right')
        plt.savefig('Original_Signal')

        f = plt.figure(figsize=(10,4))
        f.subplots_adjust(bottom=0.2)
        ax = f.add_subplot(111)
        ax.plot(np.array(test.unpack_params()).T[13], label='Original signal')
        ax.plot(np.array(noisy.unpack_params()).T[13], label='Noisy signal')
        ax.set_xlabel('Time (1/128 seconds)')
        ax.set_ylabel('Amplitude')
        legend = ax.legend(loc='upper right')
        plt.savefig('With_artifacts')

        f = plt.figure(figsize=(10,4))
        f.subplots_adjust(bottom=0.2)
        ax = f.add_subplot(111)
        ax.plot(np.array(test.unpack_params()).T[13], label='Original signal')
        ax.plot(np.array(reconstructed.unpack_params()).T[13], label='Reconstructed signal')
        ax.set_xlabel('Time (1/128 seconds)')
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

        ax.plot(mse_windows, label='Maximum Threshold', linestyle=':',color='b')

        reconstructed_windows = ExperimentorService.windows(reconstructed_dataset_avg.clone(), window_size)
        mse_windows = []

        for idx, original_window in enumerate(original_windows):
            mse = ExperimentorService.mse(original_window, reconstructed_windows[idx])
            mse_windows.append(mse)

        ax.plot(mse_windows, label='Average Threshold', linestyle=':',color='g')

        reconstructed_windows = ExperimentorService.windows(reconstructed_dataset_avg_max.clone(), window_size)
        mse_windows = []

        for idx, original_window in enumerate(original_windows):
            mse = ExperimentorService.mse(original_window, reconstructed_windows[idx])
            mse_windows.append(mse)

        ax.plot(mse_windows, label='Maximum average Threshold', linestyle=':', color='r')


        ax.set_xlabel('Window #)')
        ax.set_ylabel('Mean Squared Error')
        legend = ax.legend(loc='upper right', prop={'size':6})

        plt.savefig('Mean_Squared_Comparison')
