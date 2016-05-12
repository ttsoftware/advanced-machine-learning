import random

import numpy as np

from src.artifacts.Artificer import Artificer
from src.data.DataSet import DataSet

class ExperimentorService:

    def __init__(self):
        pass

    @staticmethod
    def split_dataset(dataset, ratio=0.8):
        training_set_size = int(np.math.floor(len(dataset) * ratio))

        training_set = DataSet(dataset[:training_set_size])
        test_set = DataSet(dataset[training_set_size:])

        return training_set, test_set

    @staticmethod
    def calibrate(dataset, window_size=40):
        dataset = dataset.clone()

        threshold_max = 0
        threshold_avg = 0
        threshold_avg_max = []
        for window in ExperimentorService.windows(dataset, window_size):

            artificer = Artificer(window)
            _, avg_eigenvalue, max_eigenvalue, _ = artificer.pca_reconstruction()

            threshold_max = max(threshold_max, max_eigenvalue)
            threshold_avg = max(threshold_avg, avg_eigenvalue)
            threshold_avg_max += [max_eigenvalue]

        threshold_avg_max = np.mean(threshold_avg_max)

        return threshold_max, threshold_avg, threshold_avg_max

    @staticmethod
    def artifactify(dataset, window_size, randomly_add_artifacts=True):
        dataset = dataset.clone()

        artifact_dataset = DataSet()
        artifact_window = []
        spike_size = (window_size // 4) * 3

        for window in ExperimentorService.windows(dataset, window_size):
            artificer = Artificer(window)
            if randomly_add_artifacts:
                decision = random.randrange(0, 2)
                if decision:
                    artifict_window = artificer.add_artifacts(spike_size)
                else:
                    artifact_window = window
            else:
                artifact_window = artificer.add_artifacts(spike_size)
            artifact_dataset += artifact_window

        if len(dataset) > len(artifact_dataset):
            amount_missing = len(dataset) - len(artifact_dataset)

            return DataSet(artifact_dataset + dataset[-amount_missing:])

        return artifact_dataset

    @staticmethod
    def pca_reconstruction(dataset, window_size, threshold):
        dataset = dataset.clone()

        reconstructed_dataset = DataSet()
        rejections = []

        for window in ExperimentorService.windows(dataset, window_size):
            artificer = Artificer(window)
            reconstructed_window, _, _, rejected = artificer.pca_reconstruction(threshold=threshold)
            reconstructed_dataset += reconstructed_window
            rejections += [rejected]

        return reconstructed_dataset, rejections

    @staticmethod
    def windows(dataset, window_size):
        return [DataSet(dataset[idx * window_size:(idx + 1) * window_size]) for idx in range(len(dataset) // window_size)]

    @staticmethod
    def mse(original, reconstructed):
        original_dataset = original.unpack_params()
        reconstructed_dataset = reconstructed.unpack_params()

        sum_window = 0

        for i in range(len(original_dataset)):
            for j in range(len(original_dataset[i])):
                sum_window += np.power(original_dataset[i][j] - reconstructed_dataset[i][j], 2)

        return sum_window / (len(original_dataset) * len(original_dataset[0]))

    @staticmethod
    def sensitivity_specificity(original, reconstructed, noisy, window_size, rejected):
        sensitivity = 0
        specificity = 0

        nb_windows = 0
        nb_no_added = 0
        nb_added = 0
        nb_no_added_no_removed = 0
        nb_no_added_removed = 0
        nb_added_no_removed = 0
        nb_added_removed = 0

        original_windows = ExperimentorService.windows(original.clone(), window_size)
        reconstructed_windows = ExperimentorService.windows(reconstructed.clone(), window_size)
        noisy_windows = ExperimentorService.windows(noisy.clone(), window_size)

        for idx, original_window in enumerate(original_windows):
            reconstructed_window = reconstructed_windows[idx]
            noisy_window = noisy_windows[idx]

            # TODO: FIIIIIIIX

            if current_original == current_noisy:
                nb_no_added += 1
                if rejected[idx]:
                    nb_no_added_removed += 1
                else:
                    nb_no_added_no_removed += 1

            else:
                nb_added += 1
                if rejected[idx]:
                    nb_added_removed += 1
                else:
                    nb_added_no_removed += 1

        sensitivity = nb_added_removed / (nb_added_removed + nb_added_no_removed)
        specificity = nb_added_no_removed / (nb_no_added_no_removed + nb_no_added_removed)

        return sensitivity, specificity
