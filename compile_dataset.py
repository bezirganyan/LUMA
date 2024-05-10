import argparse
import os

import pandas as pd

from audio_processing import add_noise_to_audio, extract_audio_deep_features, sample_audio
from text_processing import add_noise_to_text, extract_deep_text_features, sample_text
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


def generate_audio_modality(audio_csv_path, audio_test_csv, audio_data_path, features_path, compactness=0,
                            num_sampling=10,
                            add_noise_train=False, add_noise_test=False,
                            noisy_data_ratio=0.1, min_snr=3, max_snr=10):
    print("Generating audio modality")
    data = pd.read_csv(audio_csv_path, index_col=0)
    train_data = data[data['label'].isin(ID_class_labels)]
    if not os.path.exists(features_path):
        extract_audio_deep_features(audio_csv_path, audio_data_path, features_path)
    if os.path.exists(audio_test_csv):
        test_data = pd.read_csv(audio_test_csv)
    else:
        train_data, test_data = generate_test_split(train_data, test_count_per_label=100, features_path=features_path)
        test_data.to_csv(audio_test_csv)
    train_data = sample_audio(train_data, features_path, compactness=compactness, num_sampling=num_sampling)
    if add_noise_train:
        train_data = add_noise_to_audio(train_data, audio_data_path, 'noisy_audio', min_snr=min_snr, max_snr=max_snr,
                                        noisy_data_ratio=noisy_data_ratio)
    if add_noise_test:
        test_data = add_noise_to_audio(test_data, audio_data_path, 'noisy_audio', min_snr=min_snr, max_snr=max_snr,
                                       noisy_data_ratio=noisy_data_ratio)
    ood_data = data[~data['label'].isin(ID_class_labels)]
    if add_noise_test:
        ood_data = add_noise_to_audio(ood_data, audio_data_path, 'noisy_audio', min_snr=min_snr, max_snr=max_snr,
                                      noisy_data_ratio=noisy_data_ratio)
    return train_data, test_data, ood_data


def generate_text_modality(text_csv_path, text_test_csv, features_path, compactness=0, num_sampling=10,
                           add_noise_train=False, add_noise_test=False, noisy_data_ratio=0.1, noise_config=None):
    print("Generating text modality")
    data = pd.read_csv(text_csv_path, sep='\t')
    train_data = data[data['label'].isin(ID_class_labels)]
    if not os.path.exists(features_path):
        extract_deep_text_features(text_csv_path, features_path)
    train_data = sample_text(train_data, features_path, compactness=compactness, num_sampling=num_sampling)
    if os.path.exists(text_test_csv):
        test_data = pd.read_csv(text_test_csv)
    else:
        train_data, test_data = generate_test_split(train_data, test_count_per_label=100, features_path=features_path)
        test_data.to_csv(text_test_csv, sep='\t')
    if add_noise_train:
        train_data = add_noise_to_text(train_data, noisy_data_ratio=noisy_data_ratio, noise_config=noise_config)
    ood_data = data[~data['label'].isin(ID_class_labels)]
    if add_noise_test:
        test_data = add_noise_to_text(test_data, noisy_data_ratio=noisy_data_ratio, noise_config=noise_config)
        ood_data = add_noise_to_text(ood_data, noisy_data_ratio=noisy_data_ratio, noise_config=noise_config)
    return train_data, test_data, ood_data


def generate_image_modality():
    print("Generating image modality")


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str)
    args, unknown = parser.parse_known_args()
    return args, unknown


if __name__ == '__main__':
    args, unknown = parse_args()
    print(args)
    print(unknown)
    generate_audio_modality('data/audio/datalist.csv', 'data/audio/datalist_test.csv',
                            'data/audio', 'audio_features.npy')
    generate_text_modality('data/text_data.tsv', 'data/text_data_test.tsv', 'text_features.npy')
    # generate_image_modality()
