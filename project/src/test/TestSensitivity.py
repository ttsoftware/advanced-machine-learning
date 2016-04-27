from __future__ import division

import unittest

import matplotlib.pyplot as plt
import numpy as np
import random

from src.artifacts.Artificer import Artificer
from src.data.DataReader import DataReader
from src.data.DataSet import DataSet
from src.data.Normalizer import Normalizer


class TestDataSet(unittest.TestCase):
    def mean_square(pca, old):
        new_data = np.array(pca.unpack_params())
        old_data = np.array(old.unpack_params())
        sum = 0
        for i in range(len(new_data)):
            for j in range(len(new_data[i])):
                sum += np.power(new_data[i][j] - old_data[i][j], 2)
        return sum / (len(new_data) * len(new_data[i]))

    def test_max_sensitivity_specificity(self):
        # filename = '../../data/subject1_csv/eeg_200605191428_epochs/small.csv'
        filename = '../../data/emotiv/EEG_Data_filtered.csv'

        dataset = DataReader.read_data(filename, ',')
        nb_windows = 0
        nb_no_added = 0
        nb_added = 0
        nb_no_added_no_removed = 0
        nb_no_added_removed = 0
        nb_added_no_removed = 0
        nb_added_removed = 0

        mse_all_dataset = []
        mse_with_artifacts = []
        mse_without_artifacts = []

        threshold = 0
        for idx in range(len(dataset) // 40):
            current_dataset = DataSet(dataset[idx * 40:(idx + 1) * 40])

            if idx < 10:
                artificer = Artificer(current_dataset, add_artifacts=False)
                max_eigenvalue = artificer.pca_reconstruction()[1]
                threshold = max(threshold, max_eigenvalue)
            else:
                nb_windows += 1
                decision = random.randrange(0, 2)
                if decision == 0:
                    nb_no_added += 1
                    artificer = Artificer(current_dataset, add_artifacts=False)
                    rejected = artificer.pca_reconstruction(threshold)[2]
                    mse = artificer.mse()
                    mse_all_dataset.append(mse[0])
                    mse_without_artifacts.append(mse[1])

                    if rejected:
                        nb_no_added_removed += 1
                    else:
                        nb_no_added_no_removed += 1
                        # artificer.visualize()
                else:
                    nb_added += 1
                    artificer = Artificer(current_dataset, add_artifacts=True)
                    rejected = artificer.pca_reconstruction(threshold)[2]
                    mse = artificer.mse()
                    mse_all_dataset.append(mse[0])
                    mse_with_artifacts.append(mse[2])
                    mse_without_artifacts.append(mse[1])

                    if rejected:
                        nb_added_removed += 1
                    else:
                        nb_added_no_removed += 1

        artificer.visualize("max")

        print 'Number of windows without artifacts: ', nb_no_added
        print 'Number of windows with artifacts: ', nb_added

        print 'True positive: ', nb_added_removed
        print 'True negative: ', nb_no_added_no_removed
        print 'False positive: ', nb_no_added_removed
        print 'False negative: ', nb_added_no_removed

        print 'Sensitivity: ', (nb_added_removed) / (nb_added_removed + nb_added_no_removed)
        print 'Specificity: ', (nb_added_no_removed) / (nb_no_added_no_removed + nb_no_added_removed)

        print np.mean(mse_all_dataset)

    def test_avg_sensitivity_specificity(self):
        # filename = '../../data/subject1_csv/eeg_200605191428_epochs/small.csv'
        filename = '../../data/emotiv/EEG_Data_filtered.csv'

        dataset = DataReader.read_data(filename, ',')
        nb_windows = 0
        nb_no_added = 0
        nb_added = 0
        nb_no_added_no_removed = 0
        nb_no_added_removed = 0
        nb_added_no_removed = 0
        nb_added_removed = 0

        mse_windows = []
        mse_with_artifacts = []
        mse_without_artifacts = []

        threshold = 0
        for idx in range(len(dataset) // 40):
            current_dataset = DataSet(dataset[idx * 40:(idx + 1) * 40])

            if idx < 10:
                artificer = Artificer(current_dataset, add_artifacts=False)
                max_eigenvalue = artificer.pca_reconstruction()[0]
                threshold = max(threshold, max_eigenvalue)
            else:
                nb_windows += 1
                decision = random.randrange(0, 2)
                if decision == 0:
                    nb_no_added += 1
                    artificer = Artificer(current_dataset, add_artifacts=False)
                    rejected = artificer.pca_reconstruction(threshold)[2]
                    mse = artificer.mse()
                    mse_windows.append(mse)

                    if rejected:
                        nb_no_added_removed += 1
                    else:
                        nb_no_added_no_removed += 1
                        # artificer.visualize()
                else:
                    nb_added += 1
                    artificer = Artificer(current_dataset, add_artifacts=True)
                    rejected = artificer.pca_reconstruction(threshold)[2]
                    mse = artificer.mse()
                    mse_windows.append(mse)

                    if rejected:
                        nb_added_removed += 1
                    else:
                        nb_added_no_removed += 1

        print 'Number of windows without artifacts: ', nb_no_added
        print 'Number of windows with artifacts: ', nb_added

        print 'True positive: ', nb_added_removed
        print 'True negative: ', nb_no_added_no_removed
        print 'False positive: ', nb_no_added_removed
        print 'False negative: ', nb_added_no_removed

        print 'Sensitivity: ', (nb_added_removed) / (nb_added_removed + nb_added_no_removed)
        print 'Specificity: ', (nb_added_no_removed) / (nb_no_added_no_removed + nb_no_added_removed)

        print 'MSE: '
        print mse_windows

    def test_plot_mse(self):
        filename = '../../data/subject1_csv/eeg_200605191428_epochs/small.csv'

        dataset = DataReader.read_data(filename, ',')

        normalizer = Normalizer(dataset)
        dataset = normalizer.normalize_means_std(dataset)

        old_dataset = dataset.clone()

        # Add random noise to 3 randomly chosen columns
        noisy_set, noise_interval = dataset.add_artifacts()
        artifact_set = noisy_set[noise_interval[0]:noise_interval[1]]

        mses = []
        mses2 = []
        variances = []

        for variance in np.linspace(0.7, 0.99, num=5):
            print variance
            projection_dataset = noisy_set.project_pca(k=None, component_variance=variance)
            artifact_set_pca = projection_dataset[noise_interval[0]:noise_interval[1]]
            mse = self.mean_square(projection_dataset, old_dataset)
            mses.append(mse)
            mse2 = self.mean_square(DataSet(artifact_set_pca), DataSet(artifact_set))
            mses2.append(mse2)
            variances.append(variance)

        f, axarr = plt.subplots(2, 1)
        axarr[0].set_title('For all dataset')
        axarr[1].set_title('For sections with artifacts')
        axarr[0].plot(variances, mses)
        axarr[1].plot(variances, mses2)
        axarr[1].set_xlabel('Variance threshold for PCA components')
        plt.show()
