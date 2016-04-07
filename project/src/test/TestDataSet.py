import unittest
import matplotlib.pyplot as plt
import numpy as np

from src.data.DataReader import DataReader
from src.data.DataSet import DataSet
from src.data.Normalizer import Normalizer


class TestDataSet(unittest.TestCase):
    def test_pca(self):

        filename = '../../data/emotiv/EEG_Data_filtered.csv'
        filename_artifacts = '../../data/subject1_csv/eeg_200605191428_epochs/tiny_artifacts.csv'

        dataset = DataReader.read_data(filename, ',')
        dataset = DataSet(dataset[220:280])

        # Add random noise to 3 randomly chosen columns
        # noise_dataset, spike_range = dataset.add_artifacts()
        noise_dataset = dataset.clone()  # DataReader.read_data(filename_artifacts, ',')

        normalizer = Normalizer(noise_dataset)
        noise_dataset = normalizer.normalize_means(noise_dataset)

        sub_set_size = 14

        reconstructed_dataset = noise_dataset.project_pca(k=None, component_variance=0.90)
        reconstructed_dataset.add_means(normalizer.dimensions_means)

        # TODO: Project the principal components back to the original dataset

        f, axarr = plt.subplots(sub_set_size, 1)
        #axarr[0, 0].set_title('Original EEG')
        #axarr[0, 1].set_title('Normalized EEG')
        axarr[0].set_title('Corrected EEG')
        #axarr[0, 3].set_title('Principal components')

        for index, i in enumerate(range(sub_set_size)):
            #axarr[index, 0].plot(np.array(dataset.unpack_params()).T[i])
            axarr[index].plot(np.array(dataset.unpack_params()).T[i], color='r')
            #axarr[index, 1].plot(np.array(noise_dataset.unpack_params()).T[i])
            axarr[index].plot(np.array(reconstructed_dataset.unpack_params()).T[i], color='b')

        #pca_dataset_columns = np.array(projection_dataset.unpack_params()).T

        #for idx, j in enumerate(pca_dataset_columns):
        #    axarr[idx, 3].plot(j)

        # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
        #plt.setp([a.get_xticklabels() for a in axarr[0]], visible=False)
        #plt.setp([a.get_yticklabels() for a in axarr[0]], visible=True)
        plt.savefig("figure", papertype='a0', pad_inches=0, bbox_inches=0, frameon=False)
        #plt.show()
