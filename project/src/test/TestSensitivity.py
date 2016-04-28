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

        filename = '../../data/emotiv/EEG_Data_filtered.csv'
        dataset = DataReader.read_data(filename, ',')

        artifact_size = 40
        window_size = 40

        threshold_max, threshold_avg, threshold_avg_max = ExperimentorService.calibrate(dataset, window_size)

        artifact_dataset = ExperimentorService.artifactify(dataset, artifact_size, False)

        reconstructed_dataset_max, rejections_max = ExperimentorService.pca_reconstruction(artifact_dataset, window_size, threshold_max)
        reconstructed_dataset_avg, rejections_avg = ExperimentorService.pca_reconstruction(artifact_dataset, window_size, threshold_avg)
        reconstructed_dataset_avg_max, rejections_max_avg = ExperimentorService.pca_reconstruction(artifact_dataset, window_size, threshold_avg_max)

        sensitivity_max, specificity_max = ExperimentorService.sensitivity_specificity(dataset, artifact_dataset, reconstructed_dataset_max, window_size, rejections_max)
        sensitivity_avg, specificity_avg = ExperimentorService.sensitivity_specificity(dataset, artifact_dataset, reconstructed_dataset_avg, window_size, rejections_avg)
        sensitivity_avg_max, specificity_avg_max = ExperimentorService.sensitivity_specificity(dataset, artifact_dataset, reconstructed_dataset_avg_max, window_size, rejections_max_avg)

        print '--- MAX THRESHOLD ---'
        print 'Sensitivity: ', sensitivity_max
        print 'Specificity: ', specificity_max
        print '--- AVG THRESHOLD ---'
        print 'Sensitivity: ', sensitivity_avg
        print 'Specificity: ', specificity_avg
        print '--- AVG_MAX THRESHOLD ---'
        print 'Sensitivity: ', sensitivity_avg_max
        print 'Specificity: ', specificity_avg_max