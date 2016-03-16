import unittest

from src.data.DataReader import DataReader


class TestDataReader(unittest.TestCase):

    def test_read_data(self):
        filename = '../../data/subject1_csv/eeg_200605191428_epochs.csv'

        dataset = DataReader.read_data(filename, ',')

        pca_dataset = dataset.principal_component(k=None, component_variance=0.95)

        print dataset[0].params
        print pca_dataset[0].params
