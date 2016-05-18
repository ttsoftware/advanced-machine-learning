import unittest

from src.data.DataReader import DataReader
from src.experimentor.ExperimentorService import ExperimentorService
from src.visualizer.Visualizer import Visualizer
import numpy as np


class TestDataSet(unittest.TestCase):
    def test_visualize_mse_bars(self):
        filename = '../../data/emotiv/EEG_Data_filtered.csv'
        dataset = DataReader.read_data(filename, ',')

        training_set, test_set = ExperimentorService.split_dataset(dataset, ratio=0.2)

        artifact_size = 20

        artifact_dataset, _ = ExperimentorService.artifactify(test_set, artifact_size, randomly_add_artifacts=True)

        window_sizes = range(10, 300, 5)
        Visualizer.visualize_cross_validation_bars(training_set, test_set, artifact_dataset, window_sizes, name="figure_cross_validation_bars2")

    def test_visualize_mse_curves(self):
        """
        TODO: PRINT THE FUCKING BEST WINDOW SIZES FOR EACH THRESHOLD

        :return:
        """

        filename = '../../data/emotiv/EEG_Data_filtered.csv'
        dataset = DataReader.read_data(filename, ',')

        training_set, test_set = ExperimentorService.split_dataset(dataset, ratio=0.2)

        artifact_size = 60

        artifact_dataset, _ = ExperimentorService.artifactify(test_set, artifact_size, randomly_add_artifacts=True)

        window_sizes = range(5, 401, 1)
        Visualizer.visualize_cross_validation_curves(training_set, test_set, artifact_dataset, window_sizes, "figure_cross_validation_curves")

    def test_for_report(self):
        filename = '../../data/emotiv/EEG_Data_filtered.csv'
        dataset = DataReader.read_data(filename, ',')

        training_set, test_set = ExperimentorService.split_dataset(dataset, ratio=0.2)

        artifact_size = 20
        window_size = 40

        threshold_max, threshold_avg, threshold_avg_max = ExperimentorService.calibrate(training_set, window_size)

        print 'max: ' + str(threshold_max)
        print 'avg: ' + str(threshold_avg)
        print 'avg_max: ' + str(threshold_avg_max)

        artifact_dataset, _ = ExperimentorService.artifactify(test_set, artifact_size, True)

        reconstructed_dataset_max, rejections = ExperimentorService.pca_reconstruction(artifact_dataset, window_size, threshold_max)

        Visualizer.visualize_timeLine(dataset, test_set, artifact_dataset, reconstructed_dataset_max)

    def test_compare_mse(self):
        filename = '../../data/emotiv/EEG_Data_filtered.csv'
        dataset = DataReader.read_data(filename, ',')

        training_set, test_set = ExperimentorService.split_dataset(dataset, ratio=0.2)

        artifact_size = 20
        window_size = 40

        threshold_max, threshold_avg, threshold_avg_max = ExperimentorService.calibrate(training_set, window_size)

        print threshold_max
        print threshold_avg
        print threshold_avg_max

        artifact_dataset, _ = ExperimentorService.artifactify(test_set, artifact_size, True)

        reconstructed_dataset_avg, rejections = ExperimentorService.pca_reconstruction(artifact_dataset, window_size, threshold_avg)
        reconstructed_dataset_max, rejections = ExperimentorService.pca_reconstruction(artifact_dataset, window_size, threshold_max)
        reconstructed_dataset_avg_max, rejections = ExperimentorService.pca_reconstruction(artifact_dataset, window_size, threshold_avg_max)

        Visualizer.visualize_mse_on_same(test_set, reconstructed_dataset_max, reconstructed_dataset_avg, reconstructed_dataset_avg_max, window_size)



    def test_visualize_mse_bars_no_artifacts(self):
        filename = '../../data/emotiv/EEG_Data_filtered.csv'
        dataset = DataReader.read_data(filename, ',')

        training_set, test_set = ExperimentorService.split_dataset(dataset, ratio=0.2)

        window_size = 40

        threshold_max, threshold_avg, threshold_avg_max = ExperimentorService.calibrate(training_set, window_size)

        thresholds = [threshold_max, threshold_avg, threshold_avg_max]
        window_sizes = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
        Visualizer.visualize_cross_validation_bars(test_set, test_set, thresholds, window_sizes, name="figure_cross_validation_bars_no_artifacts")

    def test_visualize_mse_bars_difference(self):
        filename = '../../data/emotiv/EEG_Data_filtered.csv'
        dataset = DataReader.read_data(filename, ',')

        training_set, test_set = ExperimentorService.split_dataset(dataset, ratio=0.2)

        artifact_size = 60

        artifact_dataset, _ = ExperimentorService.artifactify(test_set, artifact_size, True)

        window_sizes = range(5, 151, 5)
        Visualizer.visualize_cross_validation_bars_percentage(training_set, test_set, artifact_dataset, window_sizes, name="figure_cross_validation_bars_difference_no_artifacts")