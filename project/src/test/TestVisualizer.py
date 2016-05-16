import random
import unittest

import numpy as np
import matplotlib.pyplot as plt

from src.data.DataReader import DataReader
from src.experimentor.ExperimentorService import ExperimentorService
from src.visualizer.Visualizer import Visualizer


class TestDataSet(unittest.TestCase):
    def test_visualize_mse_bars(self):
        filename = '../../data/emotiv/EEG_Data_filtered.csv'
        dataset = DataReader.read_data(filename, ',')

        training_set, test_set = ExperimentorService.split_dataset(dataset, ratio=0.2)

        artifact_size = 60
        window_size = 40

        threshold_max, threshold_avg, threshold_avg_max = ExperimentorService.calibrate(training_set, window_size)

        artifact_dataset = ExperimentorService.artifactify(test_set, artifact_size, False)

        # reconstructed_dataset_max, rejections = ExperimentorService.pca_reconstruction(artifact_dataset, window_size, threshold_max)
        # reconstructed_dataset_avg, rejections = ExperimentorService.pca_reconstruction(artifact_dataset, window_size, threshold_avg)
        # reconstructed_dataset_avg_max, rejections = ExperimentorService.pca_reconstruction(artifact_dataset, window_size, threshold_avg_max)

        # Visualizer.visualize_mse(test_set, reconstructed_dataset_max, window_size, 'figure_mse_max')
        # Visualizer.visualize_mse(test_set, reconstructed_dataset_avg, window_size, 'figure_mse_avg')
        # Visualizer.visualize_mse(test_set, reconstructed_dataset_avg_max, window_size, 'figure_mse_avg_max')

        thresholds = [threshold_max, threshold_avg, threshold_avg_max]
        window_sizes = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
        Visualizer.visualize_cross_validation_bars(test_set, artifact_dataset, thresholds, window_sizes, name="figure_cross_validation_bars")

    def test_visualize_mse_curves(self):
        filename = '../../data/emotiv/EEG_Data_filtered.csv'
        dataset = DataReader.read_data(filename, ',')

        training_set, test_set = ExperimentorService.split_dataset(dataset, ratio=0.2)

        artifact_size = 60
        window_size = 40

        threshold_max, threshold_avg, threshold_avg_max = ExperimentorService.calibrate(training_set, window_size)

        artifact_dataset = ExperimentorService.artifactify(test_set, artifact_size, False)

        # reconstructed_dataset_max, rejections = ExperimentorService.pca_reconstruction(artifact_dataset, window_size, threshold_max)
        # reconstructed_dataset_avg, rejections = ExperimentorService.pca_reconstruction(artifact_dataset, window_size, threshold_avg)
        # reconstructed_dataset_avg_max, rejections = ExperimentorService.pca_reconstruction(artifact_dataset, window_size, threshold_avg_max)

        # Visualizer.visualize_mse(test_set, reconstructed_dataset_max, window_size, 'figure_mse_max')
        # Visualizer.visualize_mse(test_set, reconstructed_dataset_avg, window_size, 'figure_mse_avg')
        # Visualizer.visualize_mse(test_set, reconstructed_dataset_avg_max, window_size, 'figure_mse_avg_max')

        thresholds = [threshold_max, threshold_avg, threshold_avg_max]
        window_sizes = range(5, 1001, 1)  # [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
        Visualizer.visualize_cross_validation_curves(test_set, artifact_dataset, thresholds, window_sizes, "hej")

    def test_for_report(self):
        filename = '../../data/emotiv/EEG_Data_filtered.csv'
        dataset = DataReader.read_data(filename, ',')

        training_set, test_set = ExperimentorService.split_dataset(dataset, ratio=0.2)

        artifact_size = 20
        window_size = 40

        threshold_max, threshold_avg, threshold_avg_max = ExperimentorService.calibrate(training_set, window_size)
        print threshold_max
        print threshold_avg
        artifact_dataset = ExperimentorService.artifactify(test_set, artifact_size, True)

        reconstructed_dataset_avg, rejections = ExperimentorService.pca_reconstruction(artifact_dataset, window_size, threshold_max)

        #Visualizer.visualize_all(test_set, artifact_dataset, reconstructed_dataset_avg)
        Visualizer.visualize_timeLine(test_set, artifact_dataset, reconstructed_dataset_avg, name='timeline_max')

    def test_compare_mse(self):
        filename = '../../data/emotiv/EEG_Data_filtered.csv'
        dataset = DataReader.read_data(filename, ',')

        training_set, test_set = ExperimentorService.split_dataset(dataset, ratio=0.2)

        artifact_size = 20
        window_size = 40

        threshold_max, threshold_avg, threshold_avg_max = ExperimentorService.calibrate(training_set, window_size)

        artifact_dataset = ExperimentorService.artifactify(test_set, artifact_size, True)

        reconstructed_dataset_avg, rejections = ExperimentorService.pca_reconstruction(artifact_dataset, window_size, threshold_avg)
        reconstructed_dataset_max, rejections = ExperimentorService.pca_reconstruction(artifact_dataset, window_size, threshold_max)
        reconstructed_dataset_avg_max, rejections = ExperimentorService.pca_reconstruction(artifact_dataset, window_size, threshold_avg_max)

        Visualizer.visualize_mse_on_same(test_set, reconstructed_dataset_max, reconstructed_dataset_avg, reconstructed_dataset_avg_max, window_size)