import numpy as np

from src.data.DataSet import DataSet


def project(dataset, threshold=None):
    """
    Returns a dataset that is reconstructed based on its principal components

    :param dataset: dataset to perform pca upon
    :param threshold: float The threshold for principal component variance
    :rtype : DataSet
    """

    data = np.array(dataset.unpack_params())
    data_transposed = data.T

    covariance = np.cov(data_transposed)

    # eigenvectors and eigenvalues for the from the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    W = []
    if threshold is not None:
        # Rejects all additional eigenvectors when the threshold is reached
        for idx, eigenvalue in enumerate(eigenvalues):
            if eigenvalue < threshold:
                W += [eigenvectors[idx]]
        print len(W)
    else:
        return dataset.clone(), sum(eigenvalues) / len(eigenvalues)

    W = np.array(W)

    # Projects the data onto the principal components
    eig_projection = np.empty([len(W), len(data)])
    for t, datapoint in enumerate(data):
        for q, eigenvector in enumerate(W):
            eig_projection[q][t] = sum(datapoint * eigenvector)

    # Projects the principal components back onto the data
    reconstructed_data = np.empty(data_transposed.shape)
    for j, eigen_component in enumerate(W.T):
        for t, datapoint in enumerate(eig_projection.T):
            reconstructed_data[j][t] = sum(eigen_component * datapoint)

    return DataSet(reconstructed_data.T.tolist()), sum(eigenvalues) / len(eigenvalues)
