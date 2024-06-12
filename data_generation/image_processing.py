import os

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from tqdm import tqdm

from data_generation.image_noise import brightness, contrast, defocus_blur, elastic_transform, fog, frost, \
    gaussian_noise, glass_blur, \
    impulse_noise, \
    jpeg_compression, \
    motion_blur, \
    pixelate, shot_noise, snow, zoom_blur
from data_generation.utils import download_cifar10_and_100, download_url

label_name_mapping = {'aquarium_fish': 'fish',
                      'pickup_truck': 'truck',
                      'oak_tree': 'oak',
                      'rose': 'roses',
                      'pine_tree': 'pine',
                      'palm_tree': 'palm'}

noise_mapping = {
    'gaussian_noise': gaussian_noise,
    'shot_noise': shot_noise,
    'impulse_noise': impulse_noise,
    'defocus_blur': defocus_blur,
    'frosted_glass_blur': glass_blur,
    'motion_blur': motion_blur,
    'zoom_blur': zoom_blur,
    'snow': snow,
    'fog': fog,
    'frost': frost,
    'brightness': brightness,
    'contrast': contrast,
    'elastic': elastic_transform,
    'pixelate': pixelate,
    'jpeg_compression': jpeg_compression
}


def load_cifar10(data_path):
    """
    Load the CIFAR-10 dataset
    """
    images = []
    labels = []
    if not os.path.exists(os.path.join(data_path, 'cifar-10-batches-py')):
        download_cifar10_and_100(data_path)
    for i in range(1, 6):
        data = pd.read_pickle(os.path.join(data_path, 'cifar-10-batches-py', f'data_batch_{i}'))
        image = data['data']
        label = data['labels']
        images.append(image)
        labels.append(label)
    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)
    label_names = pd.read_pickle(os.path.join(data_path, 'cifar-10-batches-py/batches.meta'))['label_names']
    for i, label in enumerate(label_names):
        label_names[i] = label_name_mapping.get(label, label)

    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    data = pd.DataFrame({'image': list(images), 'label': labels})
    data['label'] = data['label'].apply(lambda x: label_names[x])

    test_data = pd.read_pickle(os.path.join(data_path, 'cifar-10-batches-py', 'test_batch'))
    test_images = test_data['data']
    test_labels = test_data['labels']
    test_images = test_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test_data = pd.DataFrame({'image': list(test_images), 'label': test_labels})
    test_data['label'] = test_data['label'].apply(lambda x: label_names[x])
    return data, test_data, label_names


def load_cifar100(data_path):
    """
    Load the CIFAR-100 dataset
    """
    if not os.path.exists(os.path.join(data_path, 'cifar-100-python')):
        download_cifar10_and_100(data_path)
    data = pd.read_pickle(os.path.join(data_path, 'cifar-100-python', 'train'))
    label_names = pd.read_pickle(os.path.join(data_path, 'cifar-100-python', 'meta'))['fine_label_names']
    for i, label in enumerate(label_names):
        label_names[i] = label_name_mapping.get(label, label)
    images = data['data']
    labels = data['fine_labels']
    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    data = pd.DataFrame({'image': list(images), 'label': labels})
    data['label'] = data['label'].apply(lambda x: label_names[x])

    test_data = pd.read_pickle(os.path.join(data_path, 'cifar-100-python', 'test'))
    test_images = test_data['data']
    test_labels = test_data['fine_labels']
    test_images = test_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test_data = pd.DataFrame({'image': list(test_images), 'label': test_labels})
    test_data['label'] = test_data['label'].apply(lambda x: label_names[x])
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


def get_edm_generated_data(data_dir):
    data = pd.read_pickle(os.path.join(data_dir, 'edm_images.pickle'))
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
    from torch.utils.data import DataLoader

    dataset = ImageDataset(data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    # freeze the feature extractor, and unfreeze at epoch 5
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier[-1].parameters():
        param.requires_grad = True
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    for epoch in range(15):
        if epoch == 5:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
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
    model.classifier[-1] = torch.nn.Sequential(
        torch.nn.Linear(4096, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 50)
    )
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not os.path.exists(model_weight_path):
        finetune_model(data, model, model_weight_path)
    else:
        model.load_state_dict(torch.load(model_weight_path))
    model.to(device)
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


def add_noise(image, noise_config):
    convert_img = transforms.Compose([transforms.ToTensor(), transforms.ToPILImage()])
    noise_type = np.random.choice(list(noise_config.keys()))
    noise = noise_mapping[noise_type](convert_img(image), **noise_config[noise_type])
    # return image as np array
    return np.array(noise)


def add_noise_to_image(data, noise_config, output_path, noise_data_ratio=0.1, **kwargs):
    noisy_data_path = output_path
    data = data.copy()
    tqdm.pandas()
    data['image'] = data['image'].progress_apply(
        lambda x: add_noise(x, noise_config) if np.random.rand() < noise_data_ratio else x)
    data.to_pickle(noisy_data_path)
    return data


def switch_image_data_labels(data, image_features_path, switch_probability=0.1):
    """
    Randomly Switch the labels of the image data. Switching to a class that is closer in the feature space
    has higher probability.
    Parameters
    ----------
    data : pd.DataFrame
        Pandas dataframe containing the image data

    Returns
    -------
    pd.DataFrame
        Data with the labels switched
    """
    data = data.copy()
    with open(image_features_path, 'rb') as f:
        text_features = np.load(f)
    for i in tqdm(data.index.values):
        if np.random.rand() < switch_probability:
            feature = text_features[i]
            other_class_features = text_features[data[data['label'] != data.loc[i]['label']].index]
            distances = np.linalg.norm(feature - other_class_features, axis=1)
            other_class_data = data[data['label'] != data.loc[i]['label']].copy()
            other_class_data['distance'] = distances
            # mean distance to closest 5 points of each class
            to_class_distances = other_class_data.groupby('label')['distance'].nsmallest(5).groupby('label').mean()
            data.at[i, 'label'] = to_class_distances.idxmin()

    return data


if __name__ == '__main__':
    data_10, test_data_10, label_names_10 = load_cifar10()
    data_100, test_data_100, label_names_100 = load_cifar100()
    label_names = set(label_names_10).union(set(label_names_100))
    get_edm_generated_data('../data', label_names_100)
