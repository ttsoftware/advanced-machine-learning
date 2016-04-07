import numpy as np

from src.data.DataSet import DataSet


def project(dataset, threshold=0.8):
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

    # Sorts the eigenvectors based on eigenvalues, lowest to highest
    sorted_eig = map(lambda (idx, x): (x, eigenvectors[idx]), enumerate(eigenvalues))
    sorted_eig = sorted(sorted_eig, key=lambda e: e[0], reverse=False)

    eigenvaluesum = sum(eigenvalues)
    eigenvaluethreshold = eigenvaluesum * threshold

    cumsum_sorted_eig = 0
    sorted_eig_threshold_index = 0
    # Rejects all additional eigenvectors when the threshold is reached
    for i in range(len(sorted_eig)):
        if (cumsum_sorted_eig + sorted_eig[i][0]) < eigenvaluethreshold:
            cumsum_sorted_eig += sorted_eig[i][0]
        else:
            sorted_eig_threshold_index = i
            break

    W = np.array([sorted_eig[i][1] for i in range(sorted_eig_threshold_index)])

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

    return DataSet(reconstructed_data.T.tolist())
