import unittest

from src.data.DataReader import DataReader


class TestDataReader(unittest.TestCase):

    def test_read_data(self):
        filename = '../../data/subject1_csv/eeg_200605191428_epochs.csv'

        dataset = DataReader.read_data(filename, ',')

        artifacts = dataset.add_artifacts(3)

        dataset_size = len(dataset)
        dataset += artifacts

        assert len(dataset) == dataset_size + 3

        pca_dataset = dataset.principal_component(k=None, component_variance=0.95)

        print dataset[0].params
        print pca_dataset[0].params
