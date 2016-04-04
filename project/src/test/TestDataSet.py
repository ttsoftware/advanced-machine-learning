import unittest
import matplotlib.pyplot as plt
import numpy as np

from src.data.DataReader import DataReader
from src.data.DataSet import DataSet
from src.data.Normalizer import Normalizer


class TestDataSet(unittest.TestCase):
    def test_pca(self):
        filename = '../../data/subject1_csv/eeg_200605191428_epochs/small.csv'

        dataset = DataReader.read_data(filename, ',')
        dataset = DataSet(dataset[0:100])

        normalizer = Normalizer(dataset)
        dataset = normalizer.normalize_means(dataset)

        old_dataset = dataset.clone()

        # Add random noise to 3 randomly chosen columns
        noise_cols = dataset.add_artifacts()

        W, pca_dataset = dataset.principal_component(k=None, component_variance=0.80)
        projection_dataset = pca_dataset.project_pca(W)

        # TODO: Project the principal components back to the original dataset

        f, axarr = plt.subplots(10, 4)
        axarr[0, 0].set_title('Original EEG')
        axarr[0, 1].set_title('Noised EEG')
        axarr[0, 2].set_title('Corrected EEG')
        axarr[0, 3].set_title('Principal components')

        for index, i in enumerate(noise_cols):
            axarr[index, 0].plot(np.array(old_dataset.unpack_params()).T[i])
            axarr[index, 1].plot(np.array(dataset.unpack_params()).T[i])
            axarr[index, 2].plot(np.array(projection_dataset.unpack_params()).T[i])

        pca_dataset_columns = np.array(pca_dataset.unpack_params()).T

        for idx, j in enumerate(pca_dataset_columns):
            axarr[idx, 3].plot(j)

        # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
        plt.setp([a.get_xticklabels() for a in axarr[1, :]], visible=False)
        plt.setp([a.get_yticklabels() for a in axarr[:, 2]], visible=True)
        plt.show()
