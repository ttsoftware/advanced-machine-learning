import unittest

from src.data.DataReader import DataReader
from src.experimentor.ExperimentorService import ExperimentorService
from src.visualizer.Visualizer import Visualizer


class TestDataSet(unittest.TestCase):
    def test_visualize_mse(self):
        filename = '../../data/emotiv/EEG_Data_filtered.csv'
        dataset = DataReader.read_data(filename, ',')

        artifact_size = 40
        window_size = 40

        threshold_max, threshold_avg, threshold_avg_max = ExperimentorService.calibrate(dataset, window_size)

        artifact_dataset = ExperimentorService.artifactify(dataset, artifact_size, False)

        reconstructed_dataset_max, rejections = ExperimentorService.pca_reconstruction(artifact_dataset, window_size, threshold_max)
        reconstructed_dataset_avg, rejections = ExperimentorService.pca_reconstruction(artifact_dataset, window_size, threshold_avg)
        reconstructed_dataset_avg_max, rejections = ExperimentorService.pca_reconstruction(artifact_dataset, window_size, threshold_avg_max)

        Visualizer.visualize_mse(dataset, reconstructed_dataset_max, window_size, 'figure_mse_max')
        Visualizer.visualize_mse(dataset, reconstructed_dataset_avg, window_size, 'figure_mse_avg')
        Visualizer.visualize_mse(dataset, reconstructed_dataset_avg_max, window_size, 'figure_mse_avg_max')
