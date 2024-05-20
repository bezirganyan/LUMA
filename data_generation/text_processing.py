import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import numpy as np
import pandas as pd
import torch
from nltk import LancasterStemmer
from tqdm.auto import tqdm
from transformers import BertModel, BertTokenizer

from data_generation.utils import sample_class_idx

noise_dict = {
    'KeyboardNoise': nac.KeyboardAug,
    'OCRNoise': nac.OcrAug,
    'RandomCharNoise': nac.RandomCharAug,
    'AnonymNoise': naw.AntonymAug,
    'RandomWordNoise': naw.RandomWordAug,
    'SpellingNoise': naw.SpellingAug,
    'SynonymNoise': naw.SynonymAug,
    'BackTranslationNoise': naw.BackTranslationAug
}


def find_words_related_to_stem(tokens, stem, stemmer):
    return [t for t in tokens if stemmer.stem(t) == stem]


def tokenize_text(text, tokenizer):
    result = tokenizer(text, return_tensors='pt')['input_ids'][0]
    result = tokenizer.convert_ids_to_tokens(result)
    return result


def mask_label_from_text(text, label, tokenizer, stemmer):
    tokenized_text = tokenize_text(text, tokenizer)
    text = tokenizer.convert_tokens_to_ids(tokenized_text)
    text = tokenizer.decode(text, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    for related_token in find_words_related_to_stem(tokenized_text, label, stemmer):
        text = text.replace(related_token, '[MASK]')
    return text


def switch_text_data_labels(data, text_features_path, switch_probability=0.1):
    """
    Randomly Switch the labels of the text data. Switching to a class that is closer in the feature space
    has higher probability.
    Parameters
    ----------
    data : pd.DataFrame
        Pandas dataframe containing the text data

    Returns
    -------
    pd.DataFrame
        Data with the labels switched
    """
    data = data.copy()
    with open(text_features_path, 'rb') as f:
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


def extract_deep_text_features(data_csv_path, output_path='features.npy'):
    # Use Bert to extract features from text
    bert_version = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(bert_version)
    stemmer = LancasterStemmer()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BertModel.from_pretrained(bert_version, device_map=device)

    data = pd.read_csv(data_csv_path, sep='\t')
    data = data[['text', 'label']]
    features = []
    for text, label in tqdm(data.values):
        text = mask_label_from_text(text, label, tokenizer, stemmer)
        inputs = tokenizer(text, return_tensors='pt', padding=False, truncation=True).to(device)
        outputs = model(**inputs)
        features.append(outputs.last_hidden_state.cpu().detach().numpy().mean(1))
    features = np.concatenate(features, axis=0)
    print(f'Saving features to {output_path}, shape: {features.shape}')
    np.save(output_path, features)


def sample_text(data, features_path, compactness=0, num_sampling=10, n_samples_per_class=500):
    with open(features_path, 'rb') as f:
        features = np.load(f)
    sampled_data_idx = []
    for cls in data['label'].unique():
        class_data = data[data['label'] == cls]
        class_features = features[class_data.index]
        sampled_idx = sample_class_idx(class_data.index, class_features,
                                       closeness_order=compactness,
                                       num_sampling=num_sampling,
                                       samples_per_class=n_samples_per_class)
        sampled_data_idx.append(sampled_idx)
    sampled_data_idx = np.concatenate(sampled_data_idx, axis=0)
    return data.loc[sampled_data_idx]


def add_noise_to_text(data, noisy_data_ratio=0.1, noise_config=None, **kwargs):
    noises = list(noise_config.keys())
    noises = [noise_dict[noise](**noise_config[noise]) for noise in noises]
    noisy_data = data.copy()
    noisy_data['text'] = noisy_data['text'].apply(
        lambda x: noises[
            np.random.randint(0, len(noises))
        ].augment(x) if np.random.rand() < noisy_data_ratio else x)
    return noisy_data
