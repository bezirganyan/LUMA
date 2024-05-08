import os
import warnings
from pathlib import Path
from zipfile import ZipFile

import librosa
import numpy as np
import pandas as pd
import soundfile
import torch
import torchaudio
from audiomentations import AddBackgroundNoise, Normalize
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from utils import download_url, sample_class_idx


class AudioDataset(Dataset):
    def __init__(self, data_csv, base_path='data'):
        super(AudioDataset, self).__init__()
        self.data = data_csv
        self.base_path = base_path

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        path, label = tuple(self.data.iloc[idx].loc[['path', 'label']].values)
        audio, sr = torchaudio.load(os.path.join(self.base_path, path))
        return audio, sr, label


def perform_loudness_normalization(data_folder):
    data_folder = Path(data_folder)
    audio_files = []
    for audio_file in data_folder.glob('**/*.wav'):
        audio_files.append(str(audio_file))
    transform = Normalize(
        p=1.0
    )
    for audio_file in tqdm(audio_files):
        data, sr = librosa.load(audio_file)
        normalized_data = transform(data, sample_rate=sr)
        soundfile.write(audio_file, normalized_data, sr)


def extract_audio_deep_features(data_csv_path, data_path='data', output_path='features.npy'):
    bundle = torchaudio.pipelines.WAV2VEC2_LARGE
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = pd.read_csv(data_csv_path)
    if device == 'cpu':
        warnings.warn("Using CPU for feature extraction. This can be very slow. Consider using a GPU.")
    model = bundle.get_model().to(device)
    model.eval()
    dataset = AudioDataset(data, base_path=data_path)
    datloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    features = []
    for waveform, sr, label in tqdm(datloader):
        waveform = torchaudio.functional.resample(waveform, sr, bundle.sample_rate).cuda()
        feature, _ = model.extract_features(waveform.reshape(1, -1))
        features.append(feature[-1].mean(1).detach().cpu().numpy())

    features = np.concatenate(features, axis=0)
    with open(output_path, 'wb+') as f:
        np.save(f, features)


def add_noise_to_audio(data, audio_data_path, out_location, min_snr=3, max_snr=10, noisy_data_ratio=0.1):
    """
    Add noise to the audio data
    Parameters
    ----------
    data : pd.DataFrame
        Pandas dataframe containing the audio data
    audio_data_path : str
        Path to the audio data
    out_location : str
        Location to save the noisy audio data
    min_snr : int, optional, default=3
        Minimum signal-to-noise ratio
    max_snr : int, optional, default=10
        Maximum signal-to-noise ratio
    noisy_data_ratio : float, optional, default=0.1
        Ratio of the data to add noise to

    Returns
    -------
    pd.DataFrame
        Noisy audio data
    """
    data = data.copy()
    download_audio_noise_data()
    noise_files = os.listdir('esc50/ESC-50-master/audio')
    noise_paths = [os.path.join('esc50/ESC-50-master/audio', file) for file in noise_files]
    transform = AddBackgroundNoise(
        sounds_path=noise_paths,
        min_snr_db=min_snr,
        max_snr_db=max_snr,
        p=noisy_data_ratio
    )
    if not os.path.exists(out_location):
        os.makedirs(out_location)
    noisy_out_paths = []
    print("Adding noise to audio data")
    for path in tqdm(data['path']):
        cpath = os.path.join(audio_data_path, path)
        audio, sr = librosa.load(cpath)
        noisy_audio = transform(audio, sample_rate=sr)
        noisy_path = os.path.join(out_location, path)
        if not os.path.exists(os.path.dirname(noisy_path)):
            os.makedirs(os.path.dirname(noisy_path))
        soundfile.write(noisy_path, noisy_audio, sr)
        noisy_out_paths.append(noisy_path)

    data['noisy_path'] = noisy_out_paths
    return data


def download_audio_noise_data():
    if os.path.exists('esc50'):
        print("ESC-50 already exists")
        return

    print("Downloading audio noise data")
    url = 'https://github.com/karoldvl/ESC-50/archive/master.zip'
    # download the zip file if it doesn't exist
    if not os.path.exists('ESC-50.zip'):
        download_url(url, 'ESC-50.zip')
    else:
        print("ESC-50.zip already exists")
    # extract the zip file
    with ZipFile('ESC-50.zip', 'r') as zip_ref:
        zip_ref.extractall('esc50')
    print("Downloaded audio noise data")


def sample_audio(data, features_path, compactness=0, num_sampling=10, samples_per_class=600):
    """
    Sample subset from audio data
    Parameters
    ----------
    data : pd.DataFrame
        Pandas dataframe containing the audio data
    features_path: str
        Deep features path extracted from the audio data. In the file needs to be a numpy array, to be
        loaded with `np.load()`. The array should have the shape `(n_samples, n_features)`
    compactness : int or float, optional, default=0
        Compactness (how close the samples are to the mean) of the sampled data. The higher the value,
        the closer the samples are to the mean. `compactness=0` means the samples are selected uniformly at random.
    num_sampling : int, optional, default=10
        How many times sample each modality to choose the most spread out samples (in terms of total pairwise distance)
    samples_per_class : int, optional, default=600
        Number of samples to sample per class

    Returns
    -------
    pd.DataFrame
        Sampled audio data
    """
    print("Sampling audio data")
    with open(features_path, 'rb') as f:
        features = np.load(f)

    sampled_data_idx = []
    label_count = data.groupby('label')['path'].count()
    in_data_count = label_count[label_count >= samples_per_class]
    classes = in_data_count.index.values
    for cls in tqdm(classes):
        idx = data[data['label'] == cls].index.values
        sampled_idx = sample_class_idx(idx, features[idx], closeness_order=compactness, num_sampling=num_sampling,
                                       samples_per_class=samples_per_class)
        sampled_data_idx.append(sampled_idx)

    sampled_data_idx = np.concatenate(sampled_data_idx, axis=0)
    return data.iloc[sampled_data_idx]
