import os

import numpy as np
import pandas as pd

from utils import download_url

label_name_mapping = {'aquarium_fish': 'fish',
                      'pickup_truck': 'truck',
                      'oak_tree': 'oak',
                      'rose': 'roses',
                      'pine_tree': 'pine',
                      'palm_tree': 'palm'}


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
    label_names = pd.read_pickle('cifar-10-batches-py/batches.meta')['label_names']
    for i, label in enumerate(label_names):
        label_names[i] = label_name_mapping.get(label, label)

    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    data = pd.DataFrame({'image': list(images), 'label': labels})

    test_data = pd.read_pickle('cifar-10-batches-py/test_batch')
    test_images = test_data['data']
    test_labels = test_data['labels']
    test_images = test_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test_data = pd.DataFrame({'image': list(test_images), 'label': test_labels})
    return data, test_data, label_names


def load_cifar100():
    """
    Load the CIFAR-100 dataset
    """
    data = pd.read_pickle('cifar-100-python/train')
    label_names = pd.read_pickle('cifar-100-python/meta')['fine_label_names']
    for i, label in enumerate(label_names):
        label_names[i] = label_name_mapping.get(label, label)
    images = data['data']
    labels = data['fine_labels']
    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    data = pd.DataFrame({'image': list(images), 'label': labels})

    test_data = pd.read_pickle('cifar-100-python/test')
    test_images = test_data['data']
    test_labels = test_data['fine_labels']
    test_images = test_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test_data = pd.DataFrame({'image': list(test_images), 'label': test_labels})
    return data, test_data, label_names


def get_cifar_50(label_class_mapping):
    """
    Join the CIFAR-10 and CIFAR-100 datasets and sample 50 classes according to the ID_classes
    """
    data_10, test_data_10, label_names_10 = load_cifar10()
    data_100, test_data_100, label_names_100 = load_cifar100()
    data_10['label'] = data_10['label'].apply(lambda x: label_names_10[x])
    test_data_10['label'] = test_data_10['label'].apply(lambda x: label_names_10[x])
    data_100['label'] = data_100['label'].apply(lambda x: label_names_100[x])
    test_data_100['label'] = test_data_100['label'].apply(lambda x: label_names_100[x])
    data = pd.concat([data_10, data_100], axis=0)
    test_data = pd.concat([test_data_10, test_data_100], axis=0)
    return data, test_data


def get_edm_generated_data(data_dir, label_names, returned_classes, num_samples_per_class=1000):
    url = 'https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar100/1m.npz'
    if not os.path.exists(os.path.join(data_dir, 'generated_1m.npz')):
        download_url(url, os.path.join(data_dir, 'generated_1m.npz'))
    data = np.load(os.path.join(data_dir, 'generated_1m.npz'))
    images = data['image']
    labels = data['label']
    labels = [label_names[label] for label in labels]
    data = pd.DataFrame({'image': list(images), 'label': labels})
    data = data[data['label'].isin(returned_classes)]
    data = data.groupby('label').apply(lambda x: x.sample(num_samples_per_class)).reset_index(drop=True)
    return data


def extract_deep_image_features(data, output_path='data/features.npy'):
    """
    Use a pre-trained VGG16 model to extract features from images
    """
    from torchvision.models import vgg16
    import torch
    from torchvision import transforms
    from PIL import Image

    model = vgg16(pretrained=True)
    model = model.features
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    features = []
    for image in data['image']:
        image = Image.fromarray(image)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(image)
        features.append(out)
    features = torch.cat(features, dim=0)
    features = features.cpu().detach().numpy()
    # plot TSNE of features
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2)
    features = tsne.fit_transform(features)
    import matplotlib.pyplot as plt
    plt.scatter(features[:, 0], features[:, 1], c=data['label'])
    plt.show()

if __name__ == '__main__':
    data_10, test_data_10, label_names_10 = load_cifar10()
    data_100, test_data_100, label_names_100 = load_cifar100()
    label_names = set(label_names_10).union(set(label_names_100))
    get_edm_generated_data('./data', label_names_100)
