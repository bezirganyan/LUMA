import argparse
import os

import pandas as pd
from omegaconf import OmegaConf

from audio_processing import add_noise_to_audio, extract_audio_deep_features, sample_audio, switch_audio_data_labels
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
    train_data = sample_audio(train_data, features_path, **diversity_cfg)
    if sample_nosie_cfg.pop('add_noise_train', False):
        train_data = add_noise_to_audio(train_data, data_dir, audio_data_path, **sample_nosie_cfg)
    ood_data = data[~data['label'].isin(ID_class_labels)]
    if sample_nosie_cfg.pop('add_noise_test', False):
        test_data = add_noise_to_audio(test_data, data_dir, audio_data_path, 'noisy_audio', **sample_nosie_cfg)
        ood_data = add_noise_to_audio(ood_data, data_dir, audio_data_path, 'noisy_audio', **sample_nosie_cfg)
    if label_switch_prob > 0:
        train_data = switch_audio_data_labels(train_data, features_path, switch_probability=label_switch_prob)
        test_data = switch_audio_data_labels(test_data, features_path, switch_probability=label_switch_prob)
    return train_data, test_data, ood_data


def generate_text_modality(text_tsv_path, text_test_tsv, features_path, diversity_cfg, sample_nosie_cfg,
                           label_switch_prob):
    print("[*] Generating text modality")
    data = pd.read_csv(text_tsv_path, sep='\t')
    train_data = data[data['label'].isin(ID_class_labels)]
    if not os.path.exists(features_path):
        print(f'[*] Extracting deep features from {text_tsv_path}')
        extract_deep_text_features(text_tsv_path, features_path)
        print('[+] Features saved successfully!')

    print(f'[*] Sampling text data from {text_tsv_path}')
    train_data = sample_text(train_data, features_path, **diversity_cfg)
    print('[+] Text data sampled successfully!')
    if os.path.exists(text_test_tsv):
        print(f'[+] Test data found at {text_test_tsv}')
        test_data = pd.read_csv(text_test_tsv, sep='\t')
    else:
        print(f'[-] Test data not found at {text_test_tsv}')
        print(f'[*] Generating test data from {text_tsv_path}')
        train_data, test_data = generate_test_split(train_data, test_count_per_label=100, features_path=features_path)
        test_data.to_csv(text_test_tsv, sep='\t')
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


def generate_image_modality():
    print("Generating image modality")


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
    # generate_audio_modality(data_cfg.data_dir, audio_cfg.audio_csv_path, audio_cfg.audio_test_csv_path,
    #                         audio_cfg.audio_data_path,
    #                         audio_cfg.audio_features_path, diversity_cfg=audio_cfg.diversity,
    #                         sample_nosie_cfg=audio_cfg.sample_noise, label_switch_prob=audio_cfg.label_switch_prob)
    generate_text_modality(text_cfg.text_tsv_path, text_cfg.text_test_tsv_path, text_cfg.text_features_path,
                           diversity_cfg=text_cfg.diversity, sample_nosie_cfg=text_cfg.sample_noise,
                           label_switch_prob=text_cfg.label_switch_prob)
    # generate_image_modality()
