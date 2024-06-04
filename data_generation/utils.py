import os
import tarfile
import urllib.request


import numpy as np
from scipy.spatial.distance import pdist
from tqdm import tqdm


def sample_class_idx(class_idx, class_features, closeness_order=0, num_sampling=10, samples_per_class=600):
    distances = np.linalg.norm(class_features - class_features.mean(0).reshape(1, -1), axis=1) ** closeness_order
    distances = 1 / (distances + 1e-6)
    distances += 1 # to avoid less zero probabilities than necessary
    distances = distances / distances.sum()
    indexes = []
    total_distacnes = []
    for n in range(num_sampling):
        relative_class_idx = np.random.choice(range(len(class_idx)), size=samples_per_class,
                                              replace=False, p=distances.reshape(-1))
        sampled_idx = class_idx[relative_class_idx]
        sampled_features = class_features[relative_class_idx]
        sampled_distances = pdist(sampled_features).mean()
        indexes.append(sampled_idx)
        total_distacnes.append(sampled_distances)

    best = np.argmax(total_distacnes)
    return indexes[best]


def generate_test_split(data, test_count_per_label=100, features_path='features.npy', sample_count=50):
    with open(features_path, 'rb') as f:
        features = np.load(f)

    test_data = []
    for cls in data['label'].unique():
        class_data = data[data['label'] == cls]
        class_features = features[class_data.index]
        sampled_idx = sample_class_idx(class_data.index, class_features, num_sampling=sample_count,
                                       samples_per_class=test_count_per_label)
        test_data.append(sampled_idx)
    test_data = np.concatenate(test_data, axis=0)
    test_data = data.loc[test_data]
    train_data = data.drop(test_data.index)
    return train_data, test_data


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_cifar10_and_100(data_path):
    if os.path.exists(os.path.join(data_path, 'cifar-10-batches-py')):
        print("CIFAR-10 already exists")
    else:
        print("Downloading CIFAR-10")
        url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        download_url(url, os.path.join(data_path, 'cifar-10-python.tar.gz'))
        with tarfile.open(os.path.join(data_path, 'cifar-10-python.tar.gz'), 'r:gz') as tar:
            tar.extractall(path=data_path)
        print("Downloaded CIFAR-10")

    if os.path.exists(os.path.join(data_path, 'cifar-100-python')):
        print("CIFAR-100 already exists")
    else:
        print("Downloading CIFAR-100")
        url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
        download_url(url, os.path.join(data_path, 'cifar-100-python.tar.gz'))
        with tarfile.open(os.path.join(data_path, 'cifar-100-python.tar.gz'), 'r:gz') as tar:
            tar.extractall(path=data_path)
        print("Downloaded CIFAR-100")
