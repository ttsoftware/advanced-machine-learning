import unittest

from src.data.DataReader import DataReader
from src.artifacts.Artificer import Artificer
from src.data.DataSet import DataSet


class TestDataSet(unittest.TestCase):
    def test_pca(self):

        filename = '../../data/subject1_csv/eeg_200605191428_epochs/small.csv'

        dataset = DataReader.read_data(filename, ',')

        # for idx in range(len(dataset) // 40):
        #     current_dataset = DataSet(dataset[idx*40:(idx+1)*40])
        #
        #     artificer = Artificer(current_dataset)
        #     artificer.pca_reconstruction()
        #
        #     # TODO: What do we want to do with each window?

        current_dataset = DataSet(dataset[40:81])

        artificer = Artificer(current_dataset, add_artifacts=True)
        artificer.pca_reconstruction()

        artificer.visualize()
