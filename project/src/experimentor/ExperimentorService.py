import numpy as np

from src.artifacts.Artificer import Artificer
from src.data.DataSet import DataSet


class ExperimentorService:

    @staticmethod
    def calibrate(dataset, window_size=40):
        dataset = dataset.clone()

        threshold_max = 0
        threshold_avg = 0
        threshold_avg_max = 0
        for window in ExperimentorService.windows(dataset, window_size):

            artificer = Artificer(window, add_artifacts=False)
            avg_eigenvalue, max_eigenvalue, rejected = artificer.pca_reconstruction()

            threshold_max = max(threshold_max, max_eigenvalue)
            threshold_avg = max(threshold_avg, avg_eigenvalue)
            threshold_avg_max += max_eigenvalue

        threshold_avg_max = np.mean(threshold_avg_max)

        return threshold_max, threshold_avg, threshold_avg_max

    @staticmethod
    def artifactify(dataset, window_size, random=True):
        dataset = dataset.clone()

        artifact_dataset = DataSet()
        spike_size = (window_size // 4) * 3

        for window in ExperimentorService.windows(dataset, window_size):
            if random:
                decision = random.randrange(0, 2)
                if decision:
                    artificer = Artificer(window, spike_size=spike_size, add_artifacts=True)
                else:
                    artificer = Artificer(window, add_artifacts=False)
            else:
                artificer = Artificer(window, spike_size=spike_size, add_artifacts=True)
            artifact_dataset += artificer.get_noised_window()

        return artifact_dataset

    @staticmethod
    def pca_reconstruction(dataset, window_size, threshold):
        dataset = dataset.clone()

        reconstructed_dataset = DataSet()
        rejections = []

        for window in ExperimentorService.windows(dataset, window_size):
            artificer = Artificer(window, add_artifacts=False)
            threshold_max, threshold_avg, rejected = artificer.pca_reconstruction(threshold)
            reconstructed_dataset += artificer.get_reconstructed_window()
            rejections += [rejected]

        return reconstructed_dataset, rejections

    @staticmethod
    def windows(dataset, window_size):
        return [DataSet(dataset[idx * window_size:(idx + 1) * window_size]) for idx in range(len(dataset) // window_size)]
