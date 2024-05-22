import torch

from baselines.mobilenet import MobileNet
from baselines.utils import MCDropout


class ImageClassifier(torch.nn.Module):
    def __init__(self, num_classes, dropout=0.3, monte_carlo=False):
        super(ImageClassifier, self).__init__()
        # self.image_model = torch.nn.Sequential(
        #     torch.nn.Conv2d(3, 32, 3),
        #     torch.nn.ReLU(),
        #     torch.nn.MaxPool2d(2),
        #     MCDropout(dropout) if monte_carlo else torch.nn.Dropout(dropout),
        #     torch.nn.Conv2d(32, 64, 3),
        #     torch.nn.ReLU(),
        #     torch.nn.MaxPool2d(2),
        #     MCDropout(dropout) if monte_carlo else torch.nn.Dropout(dropout),
        #     torch.nn.Flatten(),
        #     torch.nn.Linear(64 * 6 * 6, num_classes),
        # )

        # self.image_model = ResNet_Cifar(BasicBlock, [3, 3, 3], num_classes, dropout, monte_carlo)
        self.image_model = MobileNet(1, num_classes)

    def forward(self, x):
        image, audio, text = x
        image = self.image_model(image)
        return image


class AudioClassifier(torch.nn.Module):
    def __init__(self, num_classes, dropout=0.5, monte_carlo=False):
        super(AudioClassifier, self).__init__()
        self.audio_model = torch.nn.Sequential(  # from batch_size x 1 x 128 x 128 spectrogram
            torch.nn.Conv2d(1, 32, 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            MCDropout(dropout) if monte_carlo else torch.nn.Dropout(dropout),
            torch.nn.Conv2d(32, 64, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            MCDropout(dropout) if monte_carlo else torch.nn.Dropout(dropout),
            torch.nn.Conv2d(64, 64, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            MCDropout(dropout) if monte_carlo else torch.nn.Dropout(dropout),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 14 * 14, num_classes),
        )

    def forward(self, x):
        image, audio, text = x
        audio = self.audio_model(audio)
        return audio


class TextClassifier(torch.nn.Module):
    def __init__(self, num_classes, dropout=0.5, monte_carlo=False):
        super(TextClassifier, self).__init__()
        self.text_model = torch.nn.Sequential(
            torch.nn.Linear(768, 512),
            torch.nn.ReLU(),
            MCDropout(dropout) if monte_carlo else torch.nn.Dropout(dropout),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            MCDropout(dropout) if monte_carlo else torch.nn.Dropout(dropout),
            torch.nn.Linear(256, num_classes),
        )

    def forward(self, x):
        image, audio, text = x
        text = self.text_model(text)
        return text


class MultimodalClassifier(torch.nn.Module):
    def __init__(self, num_classes, dropout=0.5, monte_carlo=False):
        super(MultimodalClassifier, self).__init__()
        self.image_model = ImageClassifier(num_classes, dropout, monte_carlo).image_model
        self.audio_model = AudioClassifier(num_classes, dropout, monte_carlo).audio_model
        self.text_model = TextClassifier(num_classes, dropout, monte_carlo).text_model

    def forward(self, x):
        image, audio, text = x
        image = self.image_model(image)
        audio = self.audio_model(audio)
        text = self.text_model(text)
        return (image + audio + text) / 3
