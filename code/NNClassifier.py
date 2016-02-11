from __future__ import division
import scipy.spatial.distance as spatial
from DataSet import DataSet
from DataPoint import DataPoint

class NNClassifier(object):

    def __init__(self, dataset):
        """
        :param DataSet dataset:
        """
        self.dataset = dataset.clone()

    def classify_dataset(self, k, dataset):
        """
        Set targets of all datapoints based on given k
        :param k:
        :param dataset:
        :return:
        """
        classified_set = DataSet()
        for data_point in dataset:
            target = self.classify(k, data_point.params[:])
            classified_set += [
                DataPoint(data_point.params[:], target)
            ]
        return classified_set

    def classify(self, k, coordinate, dataset=None):
        """
        Return the target with most occurences, within the k-nearest neighbours
        :param k:
        :param coordinate:
        :param dataset:
        :return:
        """
        neighbours = []

        if dataset is None:
            dataset = self.dataset

        # we expect dataset to be sorted, and we expect the first parameter to be the x-axis value.
        for i, data in enumerate(dataset):
            neighbours += [
                (data, spatial.euclidean(coordinate, data.params))
            ]

        # get targets for k-nearest neighbours
        targets = map(
            lambda x: x[0].target,
            sorted(neighbours, key=lambda x: x[1])[0:k]
        )

        # return target with most occurences
        return reduce(lambda x, y: x if targets.count(x) > y else y, targets)

    def cross_validate(self, s_fold=5, max_k=25):
        """
        Find the k from 1 to {max_k}, which yields the with best accuracy for {s_fold} validation sets.
        :param s_fold:
        :param max_k:
        :return:
        """
        s_partitions = int(len(self.dataset)/s_fold)

        test_partitions = []
        train_partitions = []

        for i in xrange(s_fold):
            start_current = i * s_partitions
            end_current = (i + 1) * s_partitions

            test_partitions += [self.dataset[start_current:end_current]]
            train_partitions += [self.dataset[:start_current] + self.dataset[end_current:]]

        best_k = (-1, -1)
        for i in range(1, max_k + 1, 2):
            accuracy = []

            for h in xrange(0, len(train_partitions)):
                for j, data in enumerate(test_partitions[h]):
                    target = self.classify(i, data.params, train_partitions[h])
                    accuracy += [target == data.target]

            if accuracy.count(True) / len(accuracy) > best_k[1]:
                best_k = (i, accuracy.count(True) / len(accuracy))

        # Return the amount of neighbors that yields the best accuracy
        return best_k[0]

    @staticmethod
    def find_accuracy(real_dataset, classified_dataset):
        """
        Find the accuracy of the classified dataset, which was classified based on classified_dataset
        :param DataSet dataset:
        :return:
        """
        return 1 - NNClassifier.find_error(real_dataset, classified_dataset)

    @staticmethod
    def find_error(real_dataset, classified_dataset):
        """
        Find the error precision of the classified dataset, which was classified based on classified_dataset
        :param DataSet dataset:
        :param k:
        :return:
        """
        error = []
        for i, data_point in enumerate(real_dataset):
            error += [data_point.target == classified_dataset[i].target]

        return error.count(False) / len(error)

    @staticmethod
    def find_sensispevity(real_dataset, classified_dataset):

        bad_count = 0
        good_count = 0

        real_bad_count = 0
        real_good_count = 0

        for i, data_point in enumerate(real_dataset):
            if classified_dataset[i].target == 0\
                    and data_point.target == 0:
                bad_count += 1

            if classified_dataset[i].target == 1\
                    and data_point.target == 1:
                good_count += 1

            if data_point.target == 0:
                real_bad_count += 1
            else:
                real_good_count += 1

        return (good_count / real_good_count), (bad_count / real_bad_count)