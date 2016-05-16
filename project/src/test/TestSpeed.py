from __future__ import division
import unittest

import time

from src.data.DataReader import DataReader
from src.experimentor.ExperimentorService import ExperimentorService


class TestSpeed(unittest.TestCase):
    def test_speed(self):
        filename = '../../data/emotiv/EEG_Data_filtered.csv'
        dataset = DataReader.read_data(filename, ',')

        training_set, test_set = ExperimentorService.split_dataset(dataset, ratio=0.2)

        artifact_size = 30
        window_size = 55

        threshold_max, threshold_avg, threshold_avg_max = ExperimentorService.calibrate(training_set, window_size)

        artifact_dataset, _ = ExperimentorService.artifactify(test_set, artifact_size, randomly_add_artifacts=False)

        start_time_max = time.time()
        reconstructed_dataset_max, _ = ExperimentorService.pca_reconstruction(artifact_dataset, window_size, threshold_max)
        end_time_max = time.time() - start_time_max

        # reconstructed_dataset_avg, _ = ExperimentorService.pca_reconstruction(artifact_dataset, window_size, threshold_avg)
        # reconstructed_dataset_avg_max, _ = ExperimentorService.pca_reconstr

        print 'We were able to reconstruct the entire test set in ' + str(end_time_max) + ' seconds.'
        print 'There are ' + str(len(ExperimentorService.windows(test_set, window_size))) + ' windows in the test set.'
        print 'On average, we can reconstruct a window in ' + str(end_time_max / len(ExperimentorService.windows(test_set, window_size))) + ' seconds.'
        print 'We can do pca projection at a rate of ' + str(1/(end_time_max / len(ExperimentorService.windows(test_set, window_size)))) + 'Hz'