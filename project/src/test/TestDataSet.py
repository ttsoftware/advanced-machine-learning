import unittest

from src.data.DataReader import DataReader
from src.artifacts.Artificer import Artificer


class TestDataSet(unittest.TestCase):
    def test_pca(self):

        filename = '../../data/subject1_csv/eeg_200605191428_epochs/small.csv'
        filename_artifacts = '../../data/subject1_csv/eeg_200605191428_epochs/tiny_artifacts.csv'

        dataset = DataReader.read_data(filename, ',')

        artificer = Artificer(dataset)
        artificer.pca_reconstruction()

        artificer.visualize()
