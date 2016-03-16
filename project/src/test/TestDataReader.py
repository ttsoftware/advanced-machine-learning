import unittest

from src.data.DataReader import DataReader


class TestDataReader(unittest.TestCase):

    def test_read_data(self):
        filename = '../../data/subject1_csv/eeg_200605191428_epochs.csv'

        dataset = DataReader.read_data(filename, ',')

        print dataset.principal_component(3)