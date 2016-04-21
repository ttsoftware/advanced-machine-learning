import unittest

from src.data.DataReader import DataReader
from src.artifacts.Artificer import Artificer
from src.data.DataSet import DataSet


class TestArtificer(unittest.TestCase):

    def test_knn_threshold(self):

        # filename = '../../data/subject1_csv/eeg_200605191428_epochs/small.csv'
        filename = '../../data/emotiv/EEG_Data_filtered.csv'

        dataset = DataReader.read_data(filename, ',')
        dataset_slice = DataSet(dataset[0:40])

        artificer = Artificer(dataset_slice, add_artifacts=True)

        threshold = artificer.knn_threshold()

        artificer.pca_reconstruction(threshold)
        artificer.visualize()
