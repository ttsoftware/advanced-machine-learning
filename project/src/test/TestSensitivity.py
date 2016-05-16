from __future__ import division
from src.experimentor.ExperimentorService import ExperimentorService

import unittest

import matplotlib.pyplot as plt
import numpy as np
import random

from src.artifacts.Artificer import Artificer
from src.data.DataReader import DataReader
from src.data.DataSet import DataSet
from src.data.Normalizer import Normalizer


class TestSensitivity(unittest.TestCase):

    def test_max_sensitivity_specificity(self):
        filename = '../../data/emotiv/EEG_Data_filtered.csv'
        dataset = DataReader.read_data(filename, ',')

        training_set, test_set = ExperimentorService.split_dataset(dataset, ratio=0.1)

        artifact_size = 20
        window_size = 40

        threshold_max, threshold_avg, threshold_avg_max = ExperimentorService.calibrate(training_set, window_size)

        artifact_test_set, artifact_list = ExperimentorService.artifactify(test_set, artifact_size, randomly_add_artifacts=True)

        reconstructed_test_set_max, rejections_max = ExperimentorService.pca_reconstruction(artifact_test_set, window_size, threshold_max)
        reconstructed_test_set_avg, rejections_avg = ExperimentorService.pca_reconstruction(artifact_test_set, window_size, threshold_avg)
        reconstructed_test_set_avg_max, rejections_max_avg = ExperimentorService.pca_reconstruction(artifact_test_set, window_size, threshold_avg_max)

        sensitivity_max, specificity_max = ExperimentorService.sensitivity_specificity(rejections_max, artifact_list)
        sensitivity_avg, specificity_avg = ExperimentorService.sensitivity_specificity(rejections_avg, artifact_list)
        sensitivity_avg_max, specificity_avg_max = ExperimentorService.sensitivity_specificity(rejections_max_avg, artifact_list)

        print '--- MAX THRESHOLD ---'
        print 'Sensitivity: ', sensitivity_max
        print 'Specificity: ', specificity_max
        print '--- AVG THRESHOLD ---'
        print 'Sensitivity: ', sensitivity_avg
        print 'Specificity: ', specificity_avg
        print '--- AVG_MAX THRESHOLD ---'
        print 'Sensitivity: ', sensitivity_avg_max
        print 'Specificity: ', specificity_avg_max

    def test_equals(self):
        filename = '../../data/emotiv/EEG_Data_filtered.csv'
        dataset1 = DataReader.read_data(filename, ',')
        dataset2 = DataReader.read_data(filename, ',')

        print dataset1 == dataset2
