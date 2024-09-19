import argparse
import os

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from data_generation.audio_processing import add_noise_to_audio, extract_audio_deep_features, sample_audio, \
    switch_audio_data_labels
from data_generation.image_processing import add_noise_to_image, extract_deep_image_features, get_edm_generated_data, \
    load_cifar10, \
    load_cifar100, switch_image_data_labels
from data_generation.text_processing import add_noise_to_text, extract_deep_text_features, sample_text, \
    switch_text_data_labels
from data_generation.utils import generate_test_split

label_class_mapping = {'man': 0, 'boy': 1, 'house': 2, 'woman': 3, 'girl': 4, 'table': 5, 'road': 6, 'horse': 7,
                       'dog': 8, 'ship': 9, 'bird': 10, 'mountain': 11, 'bed': 12, 'train': 13, 'bridge': 14,
                       'fish': 15, 'cloud': 16, 'chair': 17, 'cat': 18, 'baby': 19, 'castle': 20, 'forest': 21,
                       'television': 22, 'bear': 23, 'camel': 24, 'sea': 25, 'fox': 26, 'plain': 27, 'bus': 28,
                       'snake': 29, 'lamp': 30, 'clock': 31, 'lion': 32, 'tank': 33, 'palm': 34, 'rabbit': 35,
                       'pine': 36, 'cattle': 37, 'oak': 38, 'mouse': 39, 'frog': 40, 'ray': 41, 'bicycle': 42,
                       'truck': 43, 'elephant': 44, 'roses': 45, 'wolf': 46, 'telephone': 47, 'bee': 48, 'whale': 49}

class_label_mapping = {v: k for k, v in label_class_mapping.items()}
ID_classes = list(range(42))
ID_class_labels = [class_label_mapping[i] for i in ID_classes]


def generate_audio_modality(data_dir, audio_csv_path, audio_test_csv, audio_data_path, features_path,
                            diversity_cfg, sample_nosie_cfg, label_switch_prob):
    if sample_nosie_cfg.get('add_noise_train', False):
        if not os.path.exists(sample_nosie_cfg['output_path']):
            os.makedirs(sample_nosie_cfg['output_path'])
    print("[*] Generating audio modality")
    data = pd.read_csv(audio_csv_path)
    data['class'] = data['label'].apply(lambda x: label_class_mapping[x])
    train_data = data[data['label'].isin(ID_class_labels)]

    if not os.path.exists(features_path):
        print(f'[*] Extracting deep features from {audio_csv_path}')
        extract_audio_deep_features(audio_csv_path, audio_data_path, features_path)
        print('[+] Features saved successfully!')
    test_indices = np.loadtxt(os.path.join('split_indices', 'audio_test.txt')).astype(int)
    test_data = data.loc[test_indices]
    train_data = train_data[~train_data.index.isin(test_data.index)]
    assert set(train_data['path'].values).intersection(test_data['path'].values) == set()
    print(f'[*] Sampling audio data from {audio_csv_path}')
    if diversity_cfg.get('compactness', 0) == 0:
        train_indices = np.loadtxt(os.path.join('split_indices', 'audio_train_clean.txt')).astype(int)
        train_data = train_data.loc[train_indices]
    else:
        train_data = sample_audio(train_data, features_path, **diversity_cfg)
    print('[+] Audio data sampled successfully!')
    if sample_nosie_cfg.pop('add_noise_train', False):
        print(f'[*] Adding noise to train data')
        train_data = add_noise_to_audio(train_data, data_dir, audio_data_path, **sample_nosie_cfg)
        print('[+] Noise added to train data successfully!')
    ood_data = data[~data['label'].isin(ID_class_labels)]
    if sample_nosie_cfg.pop('add_noise_test', False):
        print(f'[*] Adding noise to test and OOD data')
        test_data = add_noise_to_audio(test_data, data_dir, audio_data_path, **sample_nosie_cfg)
        ood_data = add_noise_to_audio(ood_data, data_dir, audio_data_path, **sample_nosie_cfg)
        print('[+] Noise added to test and OOD data successfully!')
    if label_switch_prob > 0:
        print(f'[*] Switching labels of train and test data')
        train_data = switch_audio_data_labels(train_data, features_path, switch_probability=label_switch_prob)
        test_data = switch_audio_data_labels(test_data, features_path, switch_probability=label_switch_prob)
        print('[+] Labels switched successfully!')
    print('[+] Audio modality generated successfully!')
    return train_data, test_data, ood_data


def generate_text_modality(text_tsv_path, text_test_path, features_path, diversity_cfg, sample_nosie_cfg,
                           label_switch_prob):
    print("[*] Generating text modality")
    data = pd.read_csv(text_tsv_path, sep='\t')
    data['class'] = data['label'].apply(lambda x: label_class_mapping[x])
    train_data = data[data['label'].isin(ID_class_labels)]
    if not os.path.exists(features_path):
        print(f'[*] Extracting deep features from {text_tsv_path}')
        extract_deep_text_features(text_tsv_path, features_path)
        print('[+] Features saved successfully!')

    text_indices = np.loadtxt(os.path.join('split_indices', 'text_test.txt')).astype(int)
    test_data = data.loc[text_indices]
    train_data = train_data[~train_data.index.isin(test_data.index)]
    if diversity_cfg.get('compactness', 0) == 0:
        train_indices = np.loadtxt(os.path.join('split_indices', 'text_train_clean.txt')).astype(int)
        train_data = train_data.loc[train_indices]
    else:
        train_data = sample_text(train_data, features_path, **diversity_cfg, n_samples_per_class=500)
    print('[+] Text data sampled successfully!')
    if sample_nosie_cfg.pop('add_noise_train', False):
        print('[*] Adding noise to train data')
        train_data = add_noise_to_text(train_data, **sample_nosie_cfg)
        print('[+] Noise added to train data successfully!')
    ood_data = data[~data['label'].isin(ID_class_labels)]
    if sample_nosie_cfg.pop('add_noise_test', False):
        print('[*] Adding noise to test and OOD data')
        test_data = add_noise_to_text(test_data, **sample_nosie_cfg)
        ood_data = add_noise_to_text(ood_data, **sample_nosie_cfg)
        print('[+] Noise added to test and OOD data successfully!')
    if label_switch_prob > 0:
        print('[*] Switching labels of train and test data')
        train_data = switch_text_data_labels(train_data, features_path, switch_probability=label_switch_prob)
        test_data = switch_text_data_labels(test_data, features_path, switch_probability=label_switch_prob)
        print('[+] Labels switched successfully!')
    print('[+] Text modality generated successfully!')
    return train_data, test_data, ood_data


def generate_image_modality(image_data_path, image_test_path, features_path, diversity_cfg,
                            sample_nosie_cfg, label_switch_prob):
    print("[*] Generating image modality")
    if not os.path.exists(image_data_path) or not os.path.exists(image_test_path):
        print(f'[-] Image data not found at {image_data_path}')
        data, test_data = generate_cifar_50_data(image_data_path, image_test_path)
        test_data = test_data[test_data['label'].isin(label_class_mapping.keys())]
        test_data = test_data[test_data['label'].isin(ID_class_labels)].reset_index(drop=True)
        test_data['class'] = test_data['label'].apply(lambda x: label_class_mapping[x])
    else:
        data = pd.read_pickle(image_data_path)
        test_data = pd.read_pickle(image_test_path)
    data = data[data['label'].isin(label_class_mapping.keys())]
    data['class'] = data['label'].apply(lambda x: label_class_mapping[x])
    train_data = data[data['label'].isin(ID_class_labels)]
    if not features_path.endswith('.npy'):
        features_path = features_path + '.npy'
    test_features_path = features_path.replace('.npy', '_test.npy')
    if not os.path.exists(features_path):
        print(f'[*] Extracting deep features from {image_data_path}')
        train_features = extract_deep_image_features(data, features_path)
        print('[+] Features saved successfully!')
    if not os.path.exists(test_features_path):
        print(f'[*] Extracting deep features from {image_test_path}')
        test_features = extract_deep_image_features(test_data, test_features_path)
        print('[+] Features saved successfully!')

    print(f'[*] Sampling image data from {image_data_path}')
    if diversity_cfg['compactness'] == 0:
        train_data = train_data[train_data['source'] != 'synthetic100']
    assert train_data['label'].value_counts().min() >= 500
    if diversity_cfg.get('compactness', 0) == 0:
        train_indices = np.loadtxt(os.path.join('split_indices', 'image_train_clean.txt')).astype(int)
        train_data = train_data.loc[train_indices]
    else:
        train_data = sample_text(train_data, features_path, **diversity_cfg, n_samples_per_class=500)
    assert test_data['label'].value_counts().min() >= 100
    test_indices = np.loadtxt(os.path.join('split_indices', 'image_test.txt')).astype(int)
    test_data = test_data.loc[test_indices]
    print('[+] Image data sampled successfully!')
    if sample_nosie_cfg.pop('add_noise_train', False):
        print(f'[*] Adding noise to train data')
    train_data = add_noise_to_image(train_data, **sample_nosie_cfg)
    print('[+] Noise added to train data successfully!')
    ood_data = data[~data['label'].isin(ID_class_labels)]
    if sample_nosie_cfg.pop('add_noise_test', False):
        print(f'[*] Adding noise to test and OOD data')
    test_data = add_noise_to_image(test_data, **sample_nosie_cfg)
    ood_data = add_noise_to_image(ood_data, **sample_nosie_cfg)
    print('[+] Noise added to test and OOD data successfully!')
    if label_switch_prob > 0:
        print(f'[*] Switching labels of train and test data')
    train_data = switch_image_data_labels(train_data, features_path, switch_probability=label_switch_prob)
    test_data = switch_image_data_labels(test_data, test_features_path, switch_probability=label_switch_prob)
    print('[+] Labels switched successfully!')
    print('[+] Image modality generated successfully!')
    return train_data, test_data, ood_data


def generate_cifar_50_data(image_csv_path, image_test_path, data_path='./data'):
    print(f'[-] Image data not found at {image_csv_path}')
    print('[*] Loading CIFAR-50 dataset')
    data_10, test_data_10, label_names_10 = load_cifar10(data_path)
    print('[+] CIFAR-50 dataset loaded successfully!')
    print('[*] Loading CIFAR-100 dataset')
    data_100, test_data_100, label_names_100 = load_cifar100(data_path)
    print('[+] CIFAR-100 dataset loaded successfully!')
    print('[*] Loading generated CIFAR-100 dataset')
    edm_data = get_edm_generated_data(data_path)
    print('[+] Loaded generated CIFAR-100 dataset loaded successfully!')
    edm_data['source'] = 'synthetic100'
    data_100['source'] = 'cifar100'
    data_10['source'] = 'cifar10'
    data = pd.concat([data_10, data_100, edm_data], axis=0)
    data = data[data['label'].isin(label_class_mapping.keys())]
    data = data.reset_index(drop=True)

    test_data_10['source'] = 'cifar10'
    test_data_100['source'] = 'cifar100'
    test_data = pd.concat([test_data_10, test_data_100], axis=0)
    test_data = test_data[test_data['label'].isin(label_class_mapping.keys())]
    # convert images to 1 dimensional array, so that it can be saved in csv file
    data = data.reset_index(drop=True)
    data_to_save = data.copy()
    data_to_save.to_pickle(image_csv_path)
    test_data = test_data.reset_index(drop=True)
    test_data.to_pickle(image_test_path)
    data = pd.read_pickle(image_csv_path)
    assert data.equals(data_to_save)
    print('[+] Image data saved successfully!')

    return data, test_data


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='cfg/default.yml')
    args, unknown = parser.parse_known_args()
    return args, unknown


def align_data(audio, text, image, conflict=None):
    audio_chunks = []
    text_chunks = []
    image_chunks = []
    for label in ID_class_labels:
        a = audio.loc[audio['label'] == label].sample(frac=1)
        t = text.loc[text['label'] == label].sample(frac=1)
        i = image.loc[image['label'] == label].sample(frac=1)
        audio_chunks.append(a)
        text_chunks.append(t)
        image_chunks.append(i)
    if conflict is not None:
        print(f'[*] Adding conflict to {conflict*100}% of samples')
        for lab_ind, label in enumerate(ID_class_labels):
            conlict_indices = np.random.choice(range(len(audio_chunks)), int(conflict*len(audio_chunks[lab_ind])), replace=False)
            for idx in conlict_indices:
                v = np.random.randint(3)
                modality = [audio_chunks, text_chunks, image_chunks][v]
                modality[lab_ind].iloc[idx] = modality[(lab_ind + 1) % len(ID_class_labels)].iloc[idx]
                modality[lab_ind].loc[modality[lab_ind].index[idx], 'label'] = label
        print('[+] Conflict added successfully!')

    audio = pd.concat(audio_chunks, axis=0)
    text = pd.concat(text_chunks, axis=0)
    image = pd.concat(image_chunks, axis=0)
    return audio, text, image


def align_ood_data(audio, text, image):
    audio_label_counts = audio['label'].value_counts()
    text_label_counts = text['label'].value_counts()
    image_label_counts = image['label'].value_counts()
    # get the minimum label count for each class
    label_counts = pd.concat([audio_label_counts, text_label_counts, image_label_counts], axis=1).min(axis=1)
    audio_chunks = []
    text_chunks = []
    image_chunks = []
    for label in label_counts.index:
        a = audio.loc[audio['label'] == label].sample(label_counts[label]).reset_index(drop=True)
        t = text.loc[text['label'] == label].sample(label_counts[label]).reset_index(drop=True)
        i = image.loc[image['label'] == label].sample(label_counts[label]).reset_index(drop=True)
        audio_chunks.append(a)
        text_chunks.append(t)
        image_chunks.append(i)
    audio = pd.concat(audio_chunks, axis=0).reset_index(drop=True)
    text = pd.concat(text_chunks, axis=0).reset_index(drop=True)
    image = pd.concat(image_chunks, axis=0).reset_index(drop=True)
    return audio, text, image


if __name__ == '__main__':
    args, unknown = parse_args()
    # load yaml config file
    cfg = OmegaConf.load(args.cfg)
    data_cfg = cfg.data
    audio_cfg = cfg.audio
    text_cfg = cfg.text
    image_cfg = cfg.image
    # fix the seed for reproducibility
    if data_cfg.seed is not None:
        np.random.seed(data_cfg.seed)
        torch.manual_seed(data_cfg.seed)
    audio_train, audio_test, audio_ood, = generate_audio_modality(data_cfg.data_dir, audio_cfg.audio_csv_path,
                                                                  audio_cfg.audio_test_csv_path,
                                                                  audio_cfg.audio_data_path,
                                                                  audio_cfg.audio_features_path,
                                                                  diversity_cfg=audio_cfg.diversity,
                                                                  sample_nosie_cfg=audio_cfg.sample_noise,
                                                                  label_switch_prob=audio_cfg.label_switch_prob)
    if data_cfg.seed is not None:
        np.random.seed(data_cfg.seed)
        torch.manual_seed(data_cfg.seed)
    text_train, text_test, text_ood = generate_text_modality(text_cfg.text_tsv_path, text_cfg.text_test_tsv_path,
                                                             text_cfg.text_features_path,
                                                             diversity_cfg=text_cfg.diversity,
                                                             sample_nosie_cfg=text_cfg.sample_noise,
                                                             label_switch_prob=text_cfg.label_switch_prob)
    if data_cfg.seed is not None:
        np.random.seed(data_cfg.seed)
        torch.manual_seed(data_cfg.seed)
    image_train, image_test, image_ood = generate_image_modality(image_cfg.image_data_path, image_cfg.image_test_path,
                                                                 image_cfg.image_features_path,
                                                                 diversity_cfg=image_cfg.diversity,
                                                                 sample_nosie_cfg=image_cfg.sample_noise,
                                                                 label_switch_prob=image_cfg.label_switch_prob)
    if data_cfg.seed is not None:
        np.random.seed(data_cfg.seed)
        torch.manual_seed(data_cfg.seed)
    # align the data from different modalities, so that the labels are mached and shuffled(within label)
    print('Image train shape:', image_train.shape, 'Image test shape:', image_test.shape, 'Image OOD shape:')
    print('Audio train shape:', audio_train.shape, 'Audio test shape:', audio_test.shape, 'Audio OOD shape:')
    print('Text train shape:', text_train.shape, 'Text test shape:', text_test.shape, 'Text OOD shape:')
    audio_train, text_train, image_train = align_data(audio_train, text_train, image_train)
    audio_test, text_test, image_test = align_data(audio_test, text_test, image_test, conflict=data_cfg.get('conflict', None))
    audio_ood, text_ood, image_ood = align_ood_data(audio_ood, text_ood, image_ood)
    image_test['class'] = image_test['label'].apply(lambda x: label_class_mapping[x])
    assert audio_train.shape[0] == text_train.shape[0] == image_train.shape[0]
    assert audio_test.shape[0] == text_test.shape[0] == image_test.shape[0]
    assert audio_ood.shape[0] == text_ood.shape[0] == image_ood.shape[0]

    # save the data as pickle or csv or tsv file files
    audio_train.to_csv(audio_cfg.audio_train_csv_path)
    audio_test.to_csv(audio_cfg.audio_test_csv_path)
    audio_ood.to_csv(audio_cfg.audio_ood_csv_path)

    text_train.to_csv(text_cfg.text_train_tsv_path, sep='\t')
    text_test.to_csv(text_cfg.text_test_tsv_path, sep='\t')
    text_ood.to_csv(text_cfg.text_ood_tsv_path, sep='\t')

    image_train.to_pickle(image_cfg.image_train_path)
    image_test.to_pickle(image_cfg.image_test_path)
    image_ood.to_pickle(image_cfg.image_ood_path)
    print('[+] Data saved successfully!')
    print('[+] Data generation completed successfully!')
