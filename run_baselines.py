import os

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import roc_auc_score
from torchaudio.transforms import MelSpectrogram
from torchvision.transforms import ToTensor
from torchvision.transforms.v2 import Compose, Normalize

from baselines.classifiers import AudioClassifier, ImageClassifier, MultimodalClassifier, TextClassifier
from baselines.de_model import DEModel
from baselines.dirichlet import DirichletModel
from baselines.mc_model import MCDModel
from data_generation.text_processing import extract_deep_text_features
from dataset import MultiMUQDataset

if not os.path.exists('text_features_train.npy'):
    extract_deep_text_features('data/text_data_train.tsv', output_path='text_features_train.npy')
if not os.path.exists('text_features_test.npy'):
    extract_deep_text_features('data/text_data_test.tsv', output_path='text_features_test.npy')
if not os.path.exists('text_features_ood.npy'):
    extract_deep_text_features('data/text_data_ood.tsv', output_path='text_features_ood.npy')


class Text2FeatureTransform():
    def __init__(self, features_path):
        with open(features_path, 'rb') as f:
            self.features = np.load(f)

    def __call__(self, text, idx):
        return self.features[idx]


class PadCutToSizeAudioTransform():
    def __init__(self, size):
        self.size = size

    def __call__(self, audio):
        if audio.shape[-1] < self.size:
            audio = torch.nn.functional.pad(audio, (0, self.size - audio.shape[-1]))
        elif audio.shape[-1] > self.size:
            audio = audio[:, :self.size]
        return audio


train_image_path = 'data/image_data_train.pickle'
train_audio_path = 'data/audio/datalist_train.csv'
train_audio_data_path = 'data/audio'
train_text_path = 'data/text_data_train.tsv'
image_transform = Compose([
    ToTensor(),
    # Resize((224, 224)),
    Normalize(mean=(0.51, 0.49, 0.44),
              std=(0.27, 0.26, 0.28))
])
train_dataset = MultiMUQDataset(train_image_path, train_audio_path, train_audio_data_path, train_text_path,
                                text_transform=Text2FeatureTransform('text_features_train.npy'),
                                audio_transform=Compose([MelSpectrogram(), PadCutToSizeAudioTransform(128)]),
                                image_transform=image_transform)

test_image_path = 'data/image_data_test.pickle'
test_audio_path = 'data/audio/datalist_test.csv'
test_audio_data_path = 'data/audio'
test_text_path = 'data/text_data_test.tsv'

test_dataset = MultiMUQDataset(test_image_path, test_audio_path, test_audio_data_path, test_text_path,
                               text_transform=Text2FeatureTransform('text_features_test.npy'),
                               audio_transform=Compose([MelSpectrogram(), PadCutToSizeAudioTransform(128)]),
                               image_transform=image_transform)

ood_image_path = 'data/image_data_ood.pickle'
ood_audio_path = 'data/audio/datalist_ood.csv'
ood_audio_data_path = 'data/audio'
ood_text_path = 'data/text_data_ood.tsv'

ood_dataset = MultiMUQDataset(ood_image_path, ood_audio_path, ood_audio_data_path, ood_text_path,
                              text_transform=Text2FeatureTransform('text_features_ood.npy'),
                              audio_transform=Compose([MelSpectrogram(), PadCutToSizeAudioTransform(128)]),
                              image_transform=image_transform, ood=True)
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [int(0.8 * len(train_dataset)),
                                                                           len(train_dataset) - int(
                                                                               0.8 * len(train_dataset))])

batch_size = 128
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
ood_loader = torch.utils.data.DataLoader(ood_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

# Now we can use the loaders to train a model

acc_dict = {}
classes = 42
mc_samples = 100
dropout_p = 0.3
n_ensemble = 10

mc_models = [MCDModel(c, classes, mc_samples, dropout_p) for c in [ImageClassifier, AudioClassifier, TextClassifier,
                                                                   MultimodalClassifier]]
de_models = [DEModel(c, classes, n_ensemble, dropout_p) for c in [ImageClassifier, AudioClassifier, TextClassifier,
                                                                  MultimodalClassifier]]
dir_models = [DirichletModel(MultimodalClassifier, classes, dropout=dropout_p)]
models = mc_models + de_models + dir_models
for classifier in models:
    model = classifier
    try:
        model_name = classifier.__class__.__name__ + '_' + classifier.model.__class__.__name__
    except AttributeError:
        model_name = classifier.__class__.__name__ + '_' + classifier.models[0].__class__.__name__
    trainer = pl.Trainer(max_epochs=50,
                         gpus=1 if torch.cuda.is_available() else 0,
                         callbacks=[pl.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min'),
                                    pl.callbacks.ModelCheckpoint(monitor='val_loss', mode='min', save_last=True)])
    trainer.fit(model, train_loader, val_loader)
    print('Testing model')
    trainer.test(model, test_loader)
    acc_dict[model_name] = trainer.callback_metrics["test_acc"]
    acc_dict[model_name + '_ale'] = trainer.callback_metrics["test_ale"]
    acc_dict[model_name + '_entropy_ep'] = trainer.callback_metrics["test_entropy_epi"]
    aleatoric_uncertainties = model.aleatoric_uncertainties
    epistemic_uncertainties = model.epistemic_uncertainties
    print('Testing OOD')
    trainer.test(model, ood_loader)
    acc_dict[model_name + '_ood_ale'] = trainer.callback_metrics["test_ale"]
    acc_dict[model_name + '_ood'] = trainer.callback_metrics["test_acc"]
    acc_dict[model_name + '_ood_entropy_ep'] = trainer.callback_metrics["test_entropy_epi"]
    aleatoric_uncertainties_ood = model.aleatoric_uncertainties
    epistemic_uncertainties_ood = model.epistemic_uncertainties

    auc_score = roc_auc_score(
        np.concatenate([np.zeros(len(epistemic_uncertainties)), np.ones(len(epistemic_uncertainties_ood))]),
        np.concatenate([epistemic_uncertainties, epistemic_uncertainties_ood]))

    acc_dict[model_name + '_ood_auc'] = auc_score
for key, value in acc_dict.items():
    print(f'{key}: {value}')
