from DataSet import DataSet
from DataPoint import DataPoint


class DataReader(object):

    @staticmethod
    def read_data(filename, delimiter=' '):
        """
        Read our iris data, and return it in our specified data format.
        :param String filename:
        :return DataSet:
        """
        dataset = DataSet()
        with open(filename) as f:
            for line in f:
                c = line.split(delimiter)
                dataset += [DataPoint(c)]

        # we sort by x-axis so we can more easily discover nearest neighbours
        #dataset.sort()
        return dataset