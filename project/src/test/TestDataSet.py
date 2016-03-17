import unittest
import matplotlib.pyplot as plt
import numpy as np

from src.data.DataReader import DataReader


class TestDataSet(unittest.TestCase):
    def test_pca(self):
        filename = '../../data/subject1_csv/eeg_200605191428_epochs.csv'

        dataset = DataReader.read_data(filename, ',')
        # Add random noise to 3 randomly chosen columns
        noise_cols = dataset.add_artifacts(3)

        W, pca_dataset = dataset.principal_component(k=None, component_variance=0.95)

        projection_dataset = pca_dataset.project_pca(W)

        # TODO: Project the principal components back to the original dataset

        f, axarr = plt.subplots(len(noise_cols), 2)
        axarr[0, 0].set_title('Noised EEG')
        axarr[0, 1].set_title('Corrected EEG')

        for index, i in enumerate(noise_cols):
            axarr[index, 0].plot(np.array(dataset.unpack_params()).T[i])
            axarr[index, 1].plot(np.array(projection_dataset.unpack_params()).T[i])

        # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
        plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
        plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=True)
        plt.show()