import unittest

import time

from src.data.DataReader import DataReader
from src.artifacts.Artificer import Artificer
from src.data.DataSet import DataSet


class TestDataSet(unittest.TestCase):
    def test_pca(self):

        # filename = '../../data/subject1_csv/eeg_200605191428_epochs/small.csv'
        filename = '../../data/emotiv/EEG_Data_filtered.csv'

        dataset = DataReader.read_data(filename, ',')

        threshold = 0
        for idx in range(len(dataset) // 40):
            current_dataset = DataSet(dataset[idx*40:(idx+1)*40])

            if idx < 10:
                artificer = Artificer(current_dataset, add_artifacts=False)
                max_eigenvalue = artificer.pca_reconstruction()
                threshold = max(threshold, max_eigenvalue)
            else:
                artificer = Artificer(current_dataset, add_artifacts=True)
                artificer.pca_reconstruction(threshold)
                artificer.visualize()
                break

    def test_pca_speed(self):

        filename = '../../data/subject1_csv/eeg_200605191428_epochs/eeg_200605191428_epochs.csv'
        dataset = DataReader.read_data(filename, ',')

        print 'Length of dataset: ' + str(len(dataset))
        print 'Number of windows: ' + str(len(dataset) // 40)

        print 'Calculating thresholds...'

        threshold = 0
        for idx in range(25):
            current_dataset = DataSet(dataset[idx * 40:(idx + 1) * 40])

            artificer = Artificer(current_dataset, add_artifacts=False)
            max_eigenvalue, rejected = artificer.pca_reconstruction()
            threshold = max(threshold, max_eigenvalue)

        print 'Threshold: ' + str(threshold)

        print '\nTiming PCA reconstructor...'

        times = []
        for idx in range(len(dataset) // 40):
            current_dataset = DataSet(dataset[idx * 40:(idx + 1) * 40])

            start_time_constructor = time.time()
            artificer = Artificer(current_dataset, add_artifacts=False)
            midway_time = time.time() - start_time_constructor

            artificer.put_artifacts()

            start_time_reconstructor = time.time()
            artificer.pca_reconstruction(threshold)
            times.append(time.time() - start_time_reconstructor + midway_time)

        print 'Time per PCA reconstruction: ' + str(reduce(lambda x, y: x + y, times) / len(times)) + ' sec'
