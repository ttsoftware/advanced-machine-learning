import numpy as np

from src.data.DataSet import DataSet


def project(dataset, k=2, component_variance=0.8):
    """
    Returns a new dataset reduced to k principal components (dimensions)
    :param dataset: dataset to perform pca upon
    :param component_variance: float The threshold for principal component variance
    :param k:
    :rtype : DataSet
    """
    assert k < dataset.dimensions

    data = np.array(dataset.unpack_params())
    data_transposed = data.T

    covariance = np.cov(data_transposed)

    # eigenvectors and eigenvalues for the from the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    sorted_eig = map(lambda (idx, x): (x, eigenvectors[idx]), enumerate(eigenvalues))
    sorted_eig = sorted(sorted_eig, key=lambda e: e[0], reverse=False)

    if k is None:
        eigenvaluesum = sum(eigenvalues)
        eigenvaluethreshold = eigenvaluesum * component_variance

        cumsum_sorted_eig = 0
        sorted_eig_threshold_index = 0
        for i in range(len(sorted_eig)):
            if (cumsum_sorted_eig + sorted_eig[i][0]) < eigenvaluethreshold:
                cumsum_sorted_eig += sorted_eig[i][0]
            else:
                sorted_eig_threshold_index = i
                break

        W = np.array([sorted_eig[i][1] for i in range(sorted_eig_threshold_index)])
    else:
        # we choose the smallest eigenvalues
        W = np.array([sorted_eig[i][1] for i in range(k)])

    eig_projection = np.empty([len(W), len(data)])
    for t, datapoint in enumerate(data):
        for q, eigenvector in enumerate(W):
            eig_projection[q][t] = sum(datapoint * eigenvector)

    reconstructed_data = np.empty(data_transposed.shape)
    for j, eigen_component in enumerate(W.T):
        for t, datapoint in enumerate(eig_projection.T):
            reconstructed_data[j][t] = sum(eigen_component * datapoint)

    return DataSet(reconstructed_data.T.tolist())