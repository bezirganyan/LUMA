import argparse

import pandas as pd

from audio_processing import add_noise_to_audio, sample_audio


def generate_audio_modality(audio_csv_path, audio_data_path, features_path, compactness=0, num_sampling=10,
                            add_noise=False,
                            noisy_data_ratio=0.1, min_snr=3, max_snr=10):
    print("Generating audio modality")
    audio_data = pd.read_csv(audio_csv_path)
    audio_data = sample_audio(audio_data, features_path, compactness=compactness, num_sampling=num_sampling)
    if add_noise:
        audio_data = add_noise_to_audio(audio_data, audio_data_path, 'noisy_audio', min_snr=min_snr, max_snr=max_snr,
                                        noisy_data_ratio=noisy_data_ratio)
    return audio_data


def generate_text_modality():
    print("Generating text modality")


def generate_image_modality():
    print("Generating image modality")


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str)
    args, unknown = parser.parse_known_args()
    return args, unknown
