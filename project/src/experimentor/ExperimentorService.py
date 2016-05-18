from __future__ import division

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

        artifact_list = []

        artifact_dataset = DataSet()
        spike_size = (window_size // 4) * 3
        add = True

        for window in ExperimentorService.windows(dataset, window_size):
            artificer = Artificer(window)
            if randomly_add_artifacts:
                decision = random.randrange(0, 2)
                if decision and add:
                    add = False
                    artifact_window = artificer.add_artifacts(spike_size)
                    artifact_list += [1]
                else:
                    artifact_window = window
                    artifact_list += [0]
            else:
                artifact_window = artificer.add_artifacts(spike_size)
                artifact_list += [1]
            artifact_dataset += artifact_window

        if len(dataset) > len(artifact_dataset):
            amount_missing = len(dataset) - len(artifact_dataset)

            return DataSet(artifact_dataset + dataset[-amount_missing:]), artifact_list

        return artifact_dataset, artifact_list

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

        sum_window = []

        for i in range(len(original_dataset)):
            for j in range(len(original_dataset[i])):
                sum_window += [np.power(original_dataset[i][j] - reconstructed_dataset[i][j], 2)]

        return sum_window

    @staticmethod
    def difference(original, reconstructed):
        original_dataset = original.unpack_params()
        reconstructed_dataset = reconstructed.unpack_params()

        differences = []

        for i in range(len(original_dataset)):
            for j in range(len(original_dataset[i])):
                difference = abs(original_dataset[i][j] - reconstructed_dataset[i][j])
                differences += [difference / original_dataset[i][j] * 100]

        return differences

    @staticmethod
    def sensitivity_specificity(rejected_list, artifacts):

        nb_no_added = len(artifacts) - sum(artifacts)
        nb_added = sum(artifacts)
        true_negative = 0
        false_positive = 0
        false_negative = 0
        true_positive = 0

        for idx, rejected in enumerate(rejected_list):

            if rejected == artifacts[idx]:
                if rejected_list[idx]:
                    false_positive += 1
                else:
                    true_negative += 1

            else:
                if rejected_list[idx]:
                    true_positive += 1
                else:
                    false_negative += 1

        sensitivity = true_positive / (true_positive + false_negative)
        specificity = true_negative / (true_negative + false_positive)

        return sensitivity, specificity
