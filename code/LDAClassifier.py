from __future__ import division
import numpy as np
from DataPoint import DataPoint
from DataSet import DataSet


class LDAClassifier(object):

    def __init__(self, dataset):
        """
        Initialize the Linear Discriminant Analysis classifier
        :param dataset:
        """
        self.dataset = dataset.clone()
        self.classifiers = {}

        self.covariance = None
        self.parameters_count = 0
        self.class_counts = {}

        self.calc_counts()  # call before build_classifiers
        self.build_classifiers()

    def classify_dataset(self, dataset):
        classified_set = DataSet()
        for i, data_point in enumerate(dataset):
            target, score = self.classify(data_point)
            classified_set += [
                DataPoint(data_point.params[:], target)
            ]
        return classified_set

    def classify(self, data_point):
        """
        Classifies the data point using the classifiers generated from the self.dataset
        :param data_point:
        :return:
        """
        best_score = 0.0
        best_class = None

        for class_name, (posterier, mean) in self.classifiers.iteritems():
            score = self.delta_function(posterier, mean, data_point)
            if score > best_score:
                best_class = class_name
                best_score = score

        return best_class, best_score

    def calc_counts(self):
        """
        Count l_k and l
        """
        # count number of types of given parameters in each class
        for i, point in enumerate(self.dataset):
            if point.target not in self.class_counts.keys():
                self.class_counts[point.target] = 0

            self.class_counts[point.target] += 1
            self.parameters_count += 1

    def class_posterier(self, class_name):
        return self.class_counts[class_name] / self.parameters_count

    def class_mean(self, class_name):
        class_set = self.dataset.get_by_class(class_name).unpack_numpy_array()
        return (1/self.class_counts[class_name]) * sum(class_set)

    def find_covariance(self):
        z = np.zeros([self.dataset.dimensions, self.dataset.dimensions])

        for class_name, data_set in self.dataset.class_sets.iteritems():

            vectors = data_set.unpack_numpy_array()
            class_mean = self.class_mean(class_name)

            for i, vector in enumerate(vectors):
                z += np.dot((vector - class_mean), (vector - class_mean).T)

        return (1/(self.parameters_count - len(self.class_counts.keys()))) * z

    def build_classifiers(self):
        """
        Build a dictionary of delta-functions (determinant functions) for each class
        """
        self.covariance = self.find_covariance()
        for class_name, params_count in self.class_counts.iteritems():

            posterier = self.class_posterier(class_name)
            mean = self.class_mean(class_name)

            # delta-function for {class_name}
            self.classifiers[class_name] = posterier, mean

    def delta_function(self, posterier, mean, data_point):
        """
        Delta function for determining a given class associated with {posterier} and {dimension_means} for {data_point}
        :param posterier:
        :param mean:
        :param data_point:
        :return:
        """
        return float(
            np.dot(
                np.dot(
                    data_point.get_vector().T,
                    np.linalg.inv(self.covariance)
                ),
                mean
            )
            - (0.5 * np.dot(np.dot(mean.T, np.linalg.inv(self.covariance)), mean)) \
            + np.log(posterier)
        )

    @staticmethod
    def find_accuracy(real_dataset, classified_dataset):
        """
        Find the accuracy of the classified dataset, which was classified based on classified_dataset
        :param DataSet dataset:
        :return:
        """
        return 1 - LDAClassifier.find_error(real_dataset, classified_dataset)

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