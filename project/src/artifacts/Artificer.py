import numpy as np
import matplotlib.pyplot as plt

from src.data.DataSet import DataSet


def add_artifacts(dataset, k=None):
    """
    Adds k noisy artifacts to self.
    :param k:
    :return:
    """
    data = np.array(dataset.unpack_params())
    data_transposed = data.T

    # random spike interval
    # spike_range_start = randrange(0, len(rows))
    # spike_range_end = randrange(spike_range_start, (spike_range_start + len(rows)))

    spike_range_start = 20
    spike_range_end = 30

    spike_size = spike_range_end - spike_range_start

    mean = np.mean(data_transposed, axis=tuple(range(1, data_transposed.ndim)))
    cov = np.cov(data_transposed)

    # covariance matrix with smaller variance
    divisor = np.array([1 for i in range(len(cov))])
    cov_small = np.divide(cov, divisor)

    # sample from our gaussian
    samples = np.random.multivariate_normal(mean, cov_small, spike_size)

    data[spike_range_start:spike_range_end] = samples

    noise_dataset = DataSet(data.tolist())

    return noise_dataset, range(spike_range_start, spike_range_end)


def visualize(original_dataset, reconstructed_dataset, noise_dataset=None, components=10):
    if noise_dataset:
        noise_dataset = original_dataset

    f, axarr = plt.subplots(components, 3)
    axarr[0, 0].set_title('Original EEG')
    axarr[0, 1].set_title('Noised EEG')
    axarr[0, 2].set_title('Corrected EEG')

    for index, i in enumerate(range(components)):
        axarr[index, 0].plot(np.array(original_dataset.unpack_params()).T[i])
        axarr[index, 1].plot(np.array(noise_dataset.unpack_params()).T[i])
        axarr[index, 2].plot(np.array(reconstructed_dataset.unpack_params()).T[i])

    # pca_dataset_columns = np.array(projection_dataset.unpack_params()).T

    # for idx, j in enumerate(pca_dataset_columns):
    #    axarr[idx, 3].plot(j)

    # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    plt.setp([a.get_xticklabels() for a in axarr[1, :]], visible=False)
    plt.setp([a.get_yticklabels() for a in axarr[:, 2]], visible=True)
    plt.show()
