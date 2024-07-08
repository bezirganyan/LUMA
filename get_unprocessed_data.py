import argparse
import os

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from compile_dataset import generate_cifar_50_data

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


def generate_audio_modality(audio_csv_path):
    print("[*] Generating audio modality")
    data = pd.read_csv(audio_csv_path)
    data['class'] = data['label'].apply(lambda x: label_class_mapping[x])

    return data


def generate_text_modality(text_tsv_path):
    print("[*] Generating text modality")
    data = pd.read_csv(text_tsv_path, sep='\t')
    data['class'] = data['label'].apply(lambda x: label_class_mapping[x])
    return data


def generate_image_modality(image_data_path, image_test_path):
    print("[*] Generating image modality")
    if not os.path.exists(image_data_path) or not os.path.exists(image_test_path):
        print(f'[-] Image data not found at {image_data_path}')
        data, test_data = generate_cifar_50_data(image_data_path, image_test_path)
        test_data = test_data[test_data['label'].isin(label_class_mapping.keys())]
        test_data['class'] = test_data['label'].apply(lambda x: label_class_mapping[x]).reset_index(drop=True)
    else:
        data = pd.read_pickle(image_data_path)
        test_data = pd.read_pickle(image_test_path)
    train_data = data[data['label'].isin(label_class_mapping.keys())]
    test_data = test_data[test_data['label'].isin(label_class_mapping.keys())]
    data['class'] = data['label'].apply(lambda x: label_class_mapping[x])

    train_data['original_split'] = 'train'
    test_data['original_split'] = 'test'
    image = pd.concat([train_data, test_data], ignore_index=True)

    return image.reset_index(drop=True)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='cfg/unprocessed.yml')
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
    # fix the seed for reproducibility
    if data_cfg.seed is not None:
        np.random.seed(data_cfg.seed)
        torch.manual_seed(data_cfg.seed)
    audio = generate_audio_modality(audio_cfg.audio_csv_path)
    if data_cfg.seed is not None:
        np.random.seed(data_cfg.seed)
        torch.manual_seed(data_cfg.seed)
    text = generate_text_modality(text_cfg.text_tsv_path)
    if data_cfg.seed is not None:
        np.random.seed(data_cfg.seed)
        torch.manual_seed(data_cfg.seed)
    image = generate_image_modality(image_cfg.image_data_path, image_cfg.image_test_path)

    # align the data from different modalities, so that the labels are mached and shuffled(within label)
    print('Image shape:', image.shape)
    print('Audio shape:', audio.shape)
    print('Text shape:', text.shape)
    print('Image class count', image['label'].nunique())
    print('Audio class count', audio['label'].nunique())
    print('Text class count', text['label'].nunique())
    print('Image Labels Count:', image['label'].value_counts())
    print('Audio Labels Count:', audio['label'].value_counts())
    print('Text Labels Count:', text['label'].value_counts())

    # save the data as pickle or csv or tsv file files
    audio.to_csv(audio_cfg.audio_out_csv_path)
    text.to_csv(text_cfg.text_out_tsv_path, sep='\t')
    image.to_pickle(image_cfg.image_out_path)
    print('[+] Data saved successfully!')
    print('[+] Data generation completed successfully!')
