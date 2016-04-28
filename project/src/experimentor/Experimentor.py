import numpy as np

from src.artifacts.Artificer import Artificer
from src.data.DataSet import DataSet


class Experimentor:
    def __init__(self, dataset):
        self.dataset = dataset.clone()
        self.calibration_length = None

    def calibrate(self, calibration_length=10):
        self.calibration_length = calibration_length

        threshold_max = 0
        threshold_avg = 0
        threshold_avg_max = 0
        for idx in range(calibration_length):
            current_window = DataSet(self.dataset[idx * 40:(idx + 1) * 40])

            artificer = Artificer(current_window, add_artifacts=False)
            avg_eigenvalue, max_eigenvalue, rejected = artificer.pca_reconstruction()

            threshold_max = max(threshold_max, max_eigenvalue)
            threshold_avg = max(threshold_avg, avg_eigenvalue)
            threshold_avg_max += max_eigenvalue

        threshold_avg_max = np.mean(threshold_avg_max)

        return threshold_max, threshold_avg, threshold_avg_max

    def artifactify(self, random=True):
        artificers = []
        for idx in range(self.calibration_length, len(self.dataset) // 40):
            current_window = DataSet(self.dataset[idx * 40:(idx + 1) * 40])

            if random:
                decision = random.randrange(0, 2)
                if decision:
                    artificer = Artificer(current_window, True)
                else:
                    artificer = Artificer(current_window, False)
            else:
                artificer = Artificer(current_window, True)
            artificers.append(artificer)

        return artificers
