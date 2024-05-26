import torch

from baselines.utils import MCDropout


class ImageClassifier(torch.nn.Module):
    def __init__(self, num_classes, dropout=0.3, monte_carlo=False):
        super(ImageClassifier, self).__init__()
        self.image_model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            MCDropout(dropout) if monte_carlo else torch.nn.Dropout(dropout),
            torch.nn.Conv2d(32, 64, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            MCDropout(dropout) if monte_carlo else torch.nn.Dropout(dropout),
            torch.nn.Flatten(),
        )
        self.classifier = torch.nn.Linear(64 * 6 * 6, num_classes)
        self.sigma = torch.nn.Linear(64 * 6 * 6, num_classes)

    def forward(self, x):
        image, audio, text = x
        image = self.image_model(image)
        return self.classifier(image), self.sigma(image)


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
            torch.nn.Flatten()
        )
        self.classifier = torch.nn.Linear(64 * 6 * 6, num_classes)
        self.sigma = torch.nn.Linear(64 * 6 * 6, num_classes)

    def forward(self, x):
        image, audio, text = x
        audio = self.audio_model(audio)
        return self.classifier(audio), self.sigma(audio)


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
        )
        self.classifier = torch.nn.Linear(256, num_classes)
        self.sigma = torch.nn.Linear(256, num_classes)

    def forward(self, x):
        image, audio, text = x
        text = self.text_model(text)
        return self.classifier(text), self.sigma(text)


class MultimodalClassifier(torch.nn.Module):
    def __init__(self, num_classes, dropout=0.5, monte_carlo=False):
        super(MultimodalClassifier, self).__init__()
        self.image_model = ImageClassifier(num_classes, dropout, monte_carlo).image_model
        self.audio_model = AudioClassifier(num_classes, dropout, monte_carlo).audio_model
        self.text_model = TextClassifier(num_classes, dropout, monte_carlo).text_model

    def forward(self, x):
        image, audio, text = x
        image_logits, image_sigma = self.image_model(image)
        audio_logits, audio_sigma = self.audio_model(audio)
        text_logits, text_sigma = self.text_model(text)

        logits = (image_logits + audio_logits + text_logits) / 3
        sigma = (image_sigma + audio_sigma + text_sigma) / 3
        return logits, sigma
