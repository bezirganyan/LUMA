import urllib

import numpy as np
from torch import pdist
from tqdm import tqdm


def sample_class_idx(class_idx, class_features, closeness_order=0, num_sampling=10, samples_per_class=600):
    distances = np.linalg.norm(class_features - class_features.mean(0).reshape(1, -1), axis=1) ** closeness_order
    distances = 1 / distances
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


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
