import matplotlib.pyplot as plt
import numpy as np
from src.Data.DataReader import DataReader
from src.Data.Normalizer import Normalizer
from src.Data.DataSet import DataSet

def mean_square(pca, old):
   new_data = np.array(pca.unpack_params())
   old_data = np.array(old.unpack_params())
   sum = 0
   for i in range(len(new_data)):
       for j in range(len(new_data[i])):
           sum += np.power(new_data[i][j] - old_data[i][j],2)
   return sum/(len(new_data) * len(new_data[i]))

def test_pca():
    filename = '../../data/emotiv/EEG_Data_filtered.csv'
    filename_artifacts = '../../data/subject1_csv/eeg_200605191428_epochs/tiny_artifacts.csv'

    dataset = DataReader.read_data(filename, ',')
    dataset = DataSet(dataset[0:500])

    # Add random noise to 3 randomly chosen columns
    noise_dataset, spike_range = dataset.add_artifacts()
    # noise_dataset = dataset.clone()  # DataReader.read_data(filename_artifacts, ',')

    normalizer = Normalizer(noise_dataset)
    noise_dataset = normalizer.normalize_means(noise_dataset)

    sub_set_size = 14

    reconstructed_dataset = noise_dataset.project_pca(k=None, component_variance=0.90)
    reconstructed_dataset.add_means(normalizer.dimensions_means)

    noise_dataset.add_means(normalizer.dimensions_means)

    # TODO: Project the principal components back to the original dataset

    f, axarr = plt.subplots(sub_set_size, 1)
    axarr[0].set_title('Corrected EEG')

    for index, i in enumerate(range(sub_set_size)):
        axarr[index].plot(np.array(noise_dataset.unpack_params()).T[i], color='r')
        axarr[index].plot(np.array(reconstructed_dataset.unpack_params()).T[i], color='b')

    plt.show()


def plot_mse():
    filename = '../../data/subject1_csv/eeg_200605191428_epochs/small.csv'

    dataset = DataReader.read_data(filename, ',')

    normalizer = Normalizer(dataset)
    dataset = normalizer.normalize_means(dataset)

    old_dataset = dataset.clone()

    # Add random noise to 3 randomly chosen columns
    noisy_set, noise_interval = dataset.add_artifacts()
    artifact_set = noisy_set[noise_interval[0]:noise_interval[1]]

    mses = []
    mses2 = []
    variances = []

    for variance in np.linspace(0.7,0.99, num=5):
        print variance
        projection_dataset = noisy_set.project_pca(k=None, component_variance=variance)
        artifact_set_pca = projection_dataset[noise_interval[0]:noise_interval[1]]
        mse = mean_square(projection_dataset, old_dataset)
        mses.append(mse)
        mse2 = mean_square(DataSet(artifact_set_pca), DataSet(artifact_set))
        mses2.append(mse2)
        variances.append(variance)

    f, axarr = plt.subplots(2, 1)
    axarr[0].set_title('For all dataset')
    axarr[1].set_title('For sections with artifacts')
    axarr[0].plot(variances, mses)
    axarr[1].plot(variances, mses2)
    axarr[1].set_xlabel('Variance threshold for PCA components')
    plt.show()

test_pca()
