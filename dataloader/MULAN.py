
import json
import arff
import numpy as np
from skmultilearn.dataset import load_from_arff
from .config import MULAN_DIR, MULAN_INFO
from .engine import DataLoader


class MULANLoader(DataLoader):
    problem = 'ML'
    data_path = MULAN_DIR
    datasets = [
        'bibtex',
        'birds',
        'bookmarks',
        'CAL500',
        'corel5k',
        'corel5k-sparse',
        'delicious',
        'emotions',
        'enron',
        'eurlex-dc',
        'eurlex-ev',
        'eurlex-sm',
        'flags',
        'genbase',
        'mediamill',
        'medical',
        'nuswide-bow',
        'nuswide-cVLADplus',
        'rcv1subset1',
        'rcv1subset2',
        'rcv1subset3',
        'rcv1subset4',
        'rcv1subset5',
        'scene',
        'tmc2007',
        'yahoo-art',
        'yahoo-business',
        'yahoo-computer',
        'yahoo-education',
        'yahoo-entertainment',
        'yahoo-health',
        'yahoo-recreation',
        'yahoo-reference',
        'yahoo-science',
        'yahoo-social',
        'yahoo-society',
        'yeast',
    ]

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    def read_data(self, path):
        with open(MULAN_INFO) as f:
            info = json.load(f)[self.name]
            self.k = int(np.ceil(info['cardinality']))
        data, labels = load_from_arff(path, label_count=info['labels'],
                                      label_location='end')
        data = data.astype(np.float64).toarray()
        if self.name[:7] in ["nuswide", "genbase"]:
            data = data[:, 1:].astype(np.float64)
        labels = labels.toarray().astype(np.float64)
        labels *= 2
        labels -= 1
        return data, labels

    def read_trainset(self):
        path = self.data_path / self.name / (self.name + '-train.arff')
        return self.read_data(path)

    def read_testset(self):
        path = self.data_path / self.name / (self.name + '-test.arff')
        return self.read_data(path)

    @staticmethod
    def synthetic_corruption(y_train, corruption, skewed=True):
        if skewed:
            S_train = (y_train + 1) / 2
        else:
            S_train = y_train.copy()
        ind = np.random.rand(*S_train.shape) < corruption
        S_train[ind] = 0
        return S_train

    # MULAN specific methods
    @staticmethod
    def _create_train_test(path, train_path, test_path, percentage=.7):
        with open(path, 'r') as f:
            full_file = arff.load(f)
        data = full_file['data']
        nb_data = len(data)

        thres = int(percentage * nb_data)
        ind = np.random.permutation(np.arange(nb_data))
        ind_train = ind[:thres]

        train_data, test_data = [], []
        for i in range(nb_data):
            if i in ind_train:
                train_data.append(data[i])
            else:
                test_data.append(data[i])

        full_file['data'] = train_data
        with open(train_path, 'w') as f:
            arff.dump(full_file, f)

        full_file['data'] = test_data
        with open(test_path, 'w') as f:
            arff.dump(full_file, f)

    def create_train_test_set(self, percentage=.7):
        path = self.data_path / self.name / (self.name + ".arff")
        train_path = self.data_path / self.name / (self.name + "-train.arff")
        test_path = self.data_path / self.name / (self.name + "-test.arff")
        self._create_train_test(path, train_path, test_path,
                                percentage=percentage)

    def check_statistics(self):
        with open(MULAN_INFO) as f:
            info = json.load(f)[self.name]

        x_train, y_train = self.get_trainset()
        _, y_test = self.get_testset()
        nb_attributes = x_train.shape[1]
        info_nb = info['nominal'] + info['numeric']

        labels = np.vstack((y_train, y_test))
        instances = labels.shape[0]
        cardinality = labels.sum(axis=1).mean()
        density = cardinality / labels.shape[1]

        # # Fast implementation with overflow risk
        # tmp = np.tile(2 ** np.arange(labels.shape[1]), (instances, 1))
        # tmp *= labels
        # distinct = np.unique(tmp.sum(axis=1)).size

        # Slow implementaion yet avoiding overflow
        y = labels.astype(np.int).astype('U')
        tmp, i = y[:, 0], 1
        while i < y.shape[1]:
            tmp = np.core.defchararray.add(tmp, y[:, i])
            i += 1
        distinct = np.unique(tmp).size

        return {
            'instances': (instances, info['instances']),
            'nb_attributes': (nb_attributes, info_nb),
            'cardinality': (cardinality, info['cardinality']),
            'density': (density, info['density']),
            'distinct': (distinct, info['distinct'])
        }
