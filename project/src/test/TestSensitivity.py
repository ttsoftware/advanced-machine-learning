import matplotlib.pyplot as plt
import numpy as np
from src.Data.DataReader import DataReader
from src.Data.Normalizer import Normalizer

def mean_square(pca, old):
   new_data = np.array(pca.unpack_params())
   old_data = np.array(old.unpack_params())
   sum = 0
   for i in range(len(new_data)):
       for j in range(len(new_data[i])):
           sum += np.power(new_data[i][j] - old_data[i][j],2)
   return sum/(len(new_data) * len(new_data[i]))

filename = '../../data/subject1_csv/eeg_200605191428_epochs/small.csv'

dataset = DataReader.read_data(filename, ',')

normalizer = Normalizer(dataset)
dataset = normalizer.normalize_means(dataset)

old_dataset = dataset.clone()

# Add random noise to 3 randomly chosen columns
noise_cols = dataset.add_artifacts()
noise_cols = noise_cols[:10]

mses = []
variances = []

for variance in np.linspace(0.7,0.99, num=5):
    W, pca_dataset = dataset.principal_component(k=None, component_variance=variance)
    projection_dataset = pca_dataset.project_pca(W)
    mse = mean_square(projection_dataset, old_dataset)
    mses.append(mse)
    variances.append(variance)

plt.plot(variances,mses)
plt.xlabel('Variance of principal components')
plt.ylabel('Mean square error')
plt.title('For the all dataset')
plt.show()


