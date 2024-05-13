import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from tqdm import tqdm

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
        data = pd.read_pickle(f'data/cifar-10-batches-py/data_batch_{i}')
        image = data['data']
        label = data['labels']
        images.append(image)
        labels.append(label)
    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)
    label_names = pd.read_pickle('data/cifar-10-batches-py/batches.meta')['label_names']
    for i, label in enumerate(label_names):
        label_names[i] = label_name_mapping.get(label, label)

    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    data = pd.DataFrame({'image': list(images), 'label': labels})
    data['label'] = data['label'].apply(lambda x: label_names[x])

    test_data = pd.read_pickle('data/cifar-10-batches-py/test_batch')
    test_images = test_data['data']
    test_labels = test_data['labels']
    test_images = test_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test_data = pd.DataFrame({'image': list(test_images), 'label': test_labels})
    test_data['label'] = test_data['label'].apply(lambda x: label_names[x])
    return data, test_data, label_names


def load_cifar100():
    """
    Load the CIFAR-100 dataset
    """
    data = pd.read_pickle('data/cifar-100-python/train')
    label_names = pd.read_pickle('data/cifar-100-python/meta')['fine_label_names']
    for i, label in enumerate(label_names):
        label_names[i] = label_name_mapping.get(label, label)
    images = data['data']
    labels = data['fine_labels']
    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    data = pd.DataFrame({'image': list(images), 'label': labels})

    test_data = pd.read_pickle('data/cifar-100-python/test')
    test_images = test_data['data']
    test_labels = test_data['fine_labels']
    test_images = test_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test_data = pd.DataFrame({'image': list(test_images), 'label': test_labels})
    return data, test_data, label_names


def get_cifar_50():
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


class ImageDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data['image'][idx]
        image = Image.fromarray(image)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(image)
        label = self.data['class'][idx]
        return image, label

def finetune_model(data, model, output_path):
    """
    Finetune the model on the given data
    """
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from PIL import Image

    dataset = ImageDataset(data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model.classifier[-1] = torch.nn.Sequential(
        torch.nn.Linear(4096, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, len(data['label'].unique()))
    )
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    for epoch in range(5):
        model.train()
        corrects = 0
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            correct = (outputs.argmax(1) == labels).sum().item()
            corrects += correct
        loss = loss.item()
        accuracy = corrects / len(data)
        print(f'Epoch {epoch + 1}, Loss: {loss}, Accuracy: {accuracy}')
    torch.save(model.state_dict(), output_path)


def extract_deep_image_features(data, output_path='data/features.npy', model_weight_path='data/vgg11.pth'):
    """
    Use a pre-trained VGG16 model to extract features from images
    """
    from torchvision.models import vgg11_bn
    import torch
    from torchvision import transforms
    from PIL import Image

    model = vgg11_bn(pretrained=True)
    # model.classifier = model.classifier[:-1]
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    if not os.path.exists(model_weight_path):
        finetune_model(data, model, model_weight_path)
    else:
        model.load_state_dict(torch.load(model_weight_path))
    features = []
    for image in tqdm(data['image']):
        image = Image.fromarray(image)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(image)
        features.append(out.detach().cpu().numpy())
    features = np.concatenate(features, axis=0)
    with open(output_path, 'wb+') as f:
        np.save(f, features)
    return features


if __name__ == '__main__':
    data_10, test_data_10, label_names_10 = load_cifar10()
    data_100, test_data_100, label_names_100 = load_cifar100()
    label_names = set(label_names_10).union(set(label_names_100))
    get_edm_generated_data('./data', label_names_100)
