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

    def mse(self, original, reconstructed, window_size):
        original_dataset = original.unpack_params()
        reconstructed_dataset = reconstructed.unpack_params()
        mse = []
        for idx in range(len(original_dataset) // window_size):
            current_original = (original_dataset[idx * window_size:(idx + 1) * window_size])
            current_reconstructed = (reconstructed_dataset[idx * window_size:(idx + 1) * reconstructed_dataset])
            sum_window = 0


            for i in range(len(current_original)):
                for j in range(len(current_original[i])):
                    sum_window += np.power(current_original[i][j] - current_reconstructed[i][j], 2)

            mse.append(sum_window/(len(current_original) * len(current_original[0])))

        return mse

    def sensitivity_specificity(self, original, reconstructed, noisy, window_size, rejected):
        sensitivity = 0
        specificity = 0

        nb_windows = 0
        nb_no_added = 0
        nb_added = 0
        nb_no_added_no_removed = 0
        nb_no_added_removed = 0
        nb_added_no_removed = 0
        nb_added_removed = 0

        original_dataset = original.unpack_params()
        reconstructed_dataset = reconstructed.unpack_params()
        noisy_dataset = noisy.unpack_params()

        for idx in range(len(original_dataset) // window_size):
            current_original = (original_dataset[idx * window_size:(idx + 1) * window_size])
            current_reconstructed = (reconstructed_dataset[idx * window_size:(idx + 1) * reconstructed_dataset])
            current_noisy = (noisy_dataset[idx * window_size:(idx + 1) * reconstructed_dataset])

            if(current_original == current_noisy):
                nb_no_added += 1
                if(rejected[idx]):
                    nb_no_added_removed += 1
                else:
                    nb_no_added_no_removed += 1

            else:
                nb_added += 1
                if(rejected[idx]):
                    nb_added_removed += 1
                else:
                    nb_added_no_removed +=1

        sensitivity = (nb_added_removed) / (nb_added_removed + nb_added_no_removed)
        specificity = (nb_added_no_removed) / (nb_no_added_no_removed + nb_no_added_removed)

        return sensitivity, specificity
