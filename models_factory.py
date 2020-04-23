import os
from typing import Optional

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):

    def __init__(self, dimensionality: int, n_channels: int, n_classes: int, times_wider: int = 1):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(n_channels, times_wider * 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(times_wider * 32, times_wider * 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(times_wider * 32, times_wider * 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(times_wider * 64, times_wider * 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        linear_input_size = int(times_wider * 64 * (dimensionality / 4) ** 2)
        self.layer5 = nn.Sequential(
            nn.Linear(linear_input_size, times_wider * 128),
            nn.ReLU()
        )
        self.layer6 = nn.Linear(times_wider * 128, n_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.layer5(out)
        out = self.layer6(out)
        return out


class ModelsFactory:
    MODELS = {
        'SimpleCNN': SimpleCNN
    }
    MODEL_STATE_DICT = 'model_state_dict'
    INIT_WEIGHTS = 'initial_weights'

    def __init__(
            self,
            model_name: str,
            path_to_save: Optional[str] = None,
            model_path: Optional[str] = None,
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.path_to_save = path_to_save
        self.dimensionality = None
        self.n_channels = None
        self.n_classes = None
        self.model = None
        self.init_weights = None

    def init_model(self, dimensionality: int, n_channels: int, n_classes: int,):
        """Get the architecture of the model by name."""
        self.dimensionality = dimensionality
        self.n_channels = n_channels
        self.n_classes = n_classes
        model = self.MODELS[self.model_name](
            dimensionality=dimensionality,
            n_channels=n_channels,
            n_classes=n_classes,
        )
        self.init_weights = torch.nn.utils.parameters_to_vector(model.parameters())
        if self.model_path is not None:
            self._load_model_params(model=model)
        if self.do_cuda():
            model.cuda()
            self.init_weights = self.init_weights.cuda()
        self.model = model
        return model

    def _load_model_params(self, model):
        checkpoint = torch.load(self.model_path)
        model.load_state_dict(checkpoint[self.MODEL_STATE_DICT])
        self.init_weights = checkpoint[self.INIT_WEIGHTS]

    def prepare_model_for_training(self):
        self.model.train()

    def prepare_model_for_testing(self):
        self.model.eval()

    def save_model(self, name):
        path = os.path.join(self.path_to_save, name)
        torch.save(
            {
                self.MODEL_STATE_DICT: self.model.state_dict(),
                self.INIT_WEIGHTS: self.init_weights
            },
            path
        )

    @staticmethod
    def do_cuda():
        return torch.cuda.device_count() > 0
