import unittest
import matplotlib.pyplot as plt

from src.data.DataReader import DataReader


class TestDataSet(unittest.TestCase):
    def test_pca(self):
        filename = '../../data/subject1_csv/eeg_200605191428_epochs.csv'

        dataset = DataReader.read_data(filename, ',')
        # Add random noise to 3 randomly chosen columns
        dataset.add_artifacts(3)

        pca_dataset = dataset.principal_component(k=None, component_variance=0.95)

        print pca_dataset[0].params

        # TODO: Project the principal components back to the original dataset

        plt.plot(pca_dataset.unpack_params())
        plt.ylabel('Principal components')
        #plt.show()
