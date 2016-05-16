import unittest

from src.data.DataReader import DataReader
from src.experimentor.ExperimentorService import ExperimentorService


class TestDataSet(unittest.TestCase):
    filename = '../../data/emotiv/EEG_Data_filtered.csv'
    dataset = DataReader.read_data(filename, ',')

    training_set, test_set = ExperimentorService.split_dataset(dataset, ratio=0.2)

    threshold_max, threshold_avg, threshold_avgmax = ExperimentorService.calibrate(training_set, window_size=60)

    print threshold_max
    print threshold_avg
    print threshold_avgmax