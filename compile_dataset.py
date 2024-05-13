import argparse
import os

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from audio_processing import add_noise_to_audio, extract_audio_deep_features, sample_audio, switch_audio_data_labels
from image_processing import add_noise_to_image, extract_deep_image_features, get_edm_generated_data, load_cifar10, \
    load_cifar100, switch_image_data_labels
from text_processing import add_noise_to_text, extract_deep_text_features, sample_text, switch_text_data_labels
from utils import generate_test_split

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
    print("[*] Generating audio modality")
    data = pd.read_csv(audio_csv_path, index_col=0)
    data['class'] = data['label'].apply(lambda x: label_class_mapping[x])
    train_data = data[data['label'].isin(ID_class_labels)]
    if not os.path.exists(features_path):
        print(f'[*] Extracting deep features from {audio_csv_path}')
        extract_audio_deep_features(audio_csv_path, audio_data_path, features_path)
        print('[+] Features saved successfully!')
    if os.path.exists(audio_test_csv):
        print(f'[+] Test data found at {audio_test_csv}')
        test_data = pd.read_csv(audio_test_csv)
    else:
        print(f'[-] Test data not found at {audio_test_csv}')
        print(f'[*] Generating test data from {audio_csv_path}')
        train_data, test_data = generate_test_split(train_data, test_count_per_label=100, features_path=features_path)
        test_data.to_csv(audio_test_csv)
        print('[+] Test data generated successfully!')
    print(f'[*] Sampling audio data from {audio_csv_path}')
    train_data = sample_audio(train_data, features_path, **diversity_cfg)
    print('[+] Audio data sampled successfully!')
    if sample_nosie_cfg.pop('add_noise_train', False):
        print(f'[*] Adding noise to train data')
        train_data = add_noise_to_audio(train_data, data_dir, audio_data_path, **sample_nosie_cfg)
        print('[+] Noise added to train data successfully!')
    ood_data = data[~data['label'].isin(ID_class_labels)]
    if sample_nosie_cfg.pop('add_noise_test', False):
        print(f'[*] Adding noise to test and OOD data')
        test_data = add_noise_to_audio(test_data, data_dir, audio_data_path, 'noisy_audio', **sample_nosie_cfg)
        ood_data = add_noise_to_audio(ood_data, data_dir, audio_data_path, 'noisy_audio', **sample_nosie_cfg)
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

    print(f'[*] Sampling text data from {text_tsv_path}')
    train_data = sample_text(train_data, features_path, **diversity_cfg)
    print('[+] Text data sampled successfully!')
    if os.path.exists(text_test_path):
        print(f'[+] Test data found at {text_test_path}')
        test_data = pd.read_pickle(text_test_path)
    else:
        print(f'[-] Test data not found at {text_test_path}')
        print(f'[*] Generating test data from {text_tsv_path}')
        train_data, test_data = generate_test_split(train_data, test_count_per_label=100, features_path=features_path)
        test_data.to_pickle(text_test_path)
        print('[+] Test data generated successfully!')
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
    if not os.path.exists(image_data_path):
        data = generate_cifar_50_data(image_data_path)
    else:
        data = pd.read_pickle(image_data_path)
    data['class'] = data['label'].apply(lambda x: label_class_mapping[x])
    train_data = data[data['label'].isin(ID_class_labels)]
    if not os.path.exists(features_path):
        print(f'[*] Extracting deep features from {image_data_path}')
        features = extract_deep_image_features(data, features_path)
        print('[+] Features saved successfully!')
    if os.path.exists(image_test_path):
        print(f'[+] Test data found at {image_test_path}')
        test_data = pd.read_pickle(image_test_path)
    else:
        print(f'[-] Test data not found at {image_test_path}')
        print(f'[*] Generating test data from {image_data_path}')
        train_data, test_data = generate_test_split(train_data, test_count_per_label=100, features_path=features_path)
        test_data.to_pickle(image_test_path)
        print('[+] Test data generated successfully!')
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
        test_data = switch_image_data_labels(test_data, features_path, switch_probability=label_switch_prob)
        print('[+] Labels switched successfully!')


def generate_cifar_50_data(image_csv_path):
    print(f'[-] Image data not found at {image_csv_path}')
    print('[*] Loading CIFAR-50 dataset')
    data_10, test_data_10, label_names_10 = load_cifar10()
    print('[+] CIFAR-50 dataset loaded successfully!')
    print('[*] Loading CIFAR-100 dataset')
    data_100, test_data_100, label_names_100 = load_cifar100()
    print('[+] CIFAR-100 dataset loaded successfully!')
    print('[*] Loading generated CIFAR-100 dataset')
    edm_data = get_edm_generated_data('./data', label_names_100, label_class_mapping.keys())
    print('[+] Loaded generated CIFAR-100 dataset loaded successfully!')
    edm_data['source'] = 'synthetic100'
    data_100['source'] = 'cifar100'
    data_10['source'] = 'cifar10'
    data = pd.concat([data_10, data_100, edm_data], axis=0)
    data = data[data['label'].isin(label_class_mapping.keys())]
    data = data.reset_index(drop=True)
    # convert images to 1 dimensional array, so that it can be saved in csv file
    data_to_save = data.copy()
    data_to_save.to_pickle(image_csv_path)
    data = pd.read_pickle(image_csv_path)
    assert data.equals(data)
    print('[+] Image data saved successfully!')
    return data


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='cfg/default.yml')
    args, unknown = parser.parse_known_args()
    return args, unknown


if __name__ == '__main__':
    args, unknown = parse_args()
    # load yaml config file
    cfg = OmegaConf.load(args.cfg)
    data_cfg = cfg.data
    audio_cfg = cfg.audio
    text_cfg = cfg.text
    image_cfg = cfg.image
    generate_audio_modality(data_cfg.data_dir, audio_cfg.audio_csv_path, audio_cfg.audio_test_csv_path,
                            audio_cfg.audio_data_path,
                            audio_cfg.audio_features_path, diversity_cfg=audio_cfg.diversity,
                            sample_nosie_cfg=audio_cfg.sample_noise, label_switch_prob=audio_cfg.label_switch_prob)
    generate_text_modality(text_cfg.text_tsv_path, text_cfg.text_test_tsv_path, text_cfg.text_features_path,
                           diversity_cfg=text_cfg.diversity, sample_nosie_cfg=text_cfg.sample_noise,
                           label_switch_prob=text_cfg.label_switch_prob)
    generate_image_modality(image_cfg.image_data_path, image_cfg.image_test_path, image_cfg.image_features_path,
                            diversity_cfg=image_cfg.diversity,
                            sample_nosie_cfg=image_cfg.sample_noise, label_switch_prob=image_cfg.label_switch_prob)