import unittest

import numpy as np

from src.data.DataReader import DataReader
from src.data.Normalizer import Normalizer
import src.artifacts.pca.PcaProjector as PCA
import src.artifacts.Artificer as Artificer


class TestDataSet(unittest.TestCase):
    def test_pca(self):

        filename = '../../data/subject1_csv/eeg_200605191428_epochs/small.csv'
        filename_artifacts = '../../data/subject1_csv/eeg_200605191428_epochs/tiny_artifacts.csv'

        dataset = DataReader.read_data(filename, ',')

        # Add random noise to 3 randomly chosen columns
        # noise_dataset, spike_range = Artificer.add_artifacts(dataset)
        noise_dataset = dataset.clone()  # DataReader.read_data(filename_artifacts, ',')

        normalizer = Normalizer(noise_dataset)
        noise_dataset = normalizer.subtract_means(noise_dataset)

        reconstructed_dataset = PCA.project(noise_dataset, k=None, component_variance=0.50)
        reconstructed_dataset = normalizer.add_means(reconstructed_dataset)

        Artificer.visualize(dataset, noise_dataset, reconstructed_dataset)
