import unittest

from src.data.DataReader import DataReader
from src.artifacts.Artificer import Artificer
from src.data.DataSet import DataSet


class TestDataSet(unittest.TestCase):
    def test_pca(self):

        #filename = '../../data/subject1_csv/eeg_200605191428_epochs/small.csv'
        filename = '../../data/emotiv/EEG_Data_filtered.csv'

        dataset = DataReader.read_data(filename, ',')

        threshold = 0
        for idx in range(len(dataset) // 40):
            current_dataset = DataSet(dataset[idx*40:(idx+1)*40])

            artificer = Artificer(current_dataset, add_artifacts=(True if idx > 10 else False))
            max_eigenvalue = artificer.pca_reconstruction(threshold if idx > 10 else None)

            threshold = max(threshold, max_eigenvalue)

            if idx > 10:
                artificer.visualize()
                break
