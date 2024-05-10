import numpy as np
import pandas as pd


def load_cifar10():
    """
    Load the CIFAR-10 dataset
    """
    images = []
    labels = []
    for i in range(1, 6):
        data = pd.read_pickle(f'cifar-10-batches-py/data_batch_{i}')
        image = data['data']
        label = data['labels']
        images.append(image)
        labels.append(label)
    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)

    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    data = pd.DataFrame({'image': list(images), 'label': labels})

    test_data = pd.read_pickle('cifar-10-batches-py/test_batch')
    test_images = test_data['data']
    test_labels = test_data['labels']
    test_images = test_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test_data = pd.DataFrame({'image': list(test_images), 'label': test_labels})
    return data, test_data


def load_cifar100():
    """
    Load the CIFAR-100 dataset
    """
    data = pd.read_pickle('cifar-100-python/train')
    images = data['data']
    labels = data['fine_labels']
    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    data = pd.DataFrame({'image': list(images), 'label': labels})

    test_data = pd.read_pickle('cifar-100-python/test')
    test_images = test_data['data']
    test_labels = test_data['fine_labels']
    test_images = test_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test_data = pd.DataFrame({'image': list(test_images), 'label': test_labels})
    return data, test_data


def join_cifar10_and_100():
    """
    Join the CIFAR-10 and CIFAR-100 datasets
    """
    cifar10, test_cifar10 = load_cifar10()
    cifar100, test_cifar100 = load_cifar100()
    cifar10['label'] = cifar10['label'] + 10
    test_cifar10['label'] = test_cifar10['label'] + 10

if __name__ == '__main__':
    data = load_cifar10()
    print(data.head())
    print(data.shape)
