import unittest

from src.data.DataReader import DataReader
from src.visualizer.Visualizer import Visualizer


class TestDataSet(unittest.TestCase):
    def test_visualize_mse(self):
        filename = '../../data/emotiv/EEG_Data_filtered.csv'
        dataset = DataReader.read_data(filename, ',')

        visualizer = Visualizer(dataset)
        # visualizer.visualize_mse(10, "max", 40)
        # visualizer.visualize_mse(10, "avg", 40)
        # visualizer.visualize_mse(10, "avg_max", 40)

        visualizer.visualize_cross_validation(10, [20, 30, 40, 50, 60])
