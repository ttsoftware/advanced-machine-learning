import unittest

from src.data.DataReader import DataReader


class TestDataSet(unittest.TestCase):

    def test_pca(self):
        filename = '../../data/subject1_csv/eeg_200605191428_epochs.csv'

        dataset = DataReader.read_data(filename, ',')
        dataset_size = len(dataset)

        dataset.add_artifacts(3)

        assert len(dataset) == dataset_size + 3

        pca_dataset = dataset.principal_component(k=None, component_variance=0.95)

        print dataset[0].params
        print pca_dataset[0].params
