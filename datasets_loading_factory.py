import torch
from torchvision import datasets, transforms

CIFAR_NORMALIZATION = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

DEFAULT_CIFAR_TRAIN_TRANSFORMS = transforms.Compose(
    [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*CIFAR_NORMALIZATION),
    ]
)


class DatasetGeneratorPyTorch:
    def __init__(self):
        self.dataset = None
        self.dimensionality = None
        self.n_channels = None
        self.n_classes = None

    def init_dataset(self):
        dataset = datasets.CIFAR10(
            root='./data/',
            train=True,
            download=True,
            transform=DEFAULT_CIFAR_TRAIN_TRANSFORMS,
        )
        self.dataset = dataset
        self.dimensionality = 32
        self.n_channels = 3
        self.n_classes = 10
        return dataset

    def get_data_loader(self, batch_size, shuffle=True):
        data_loader = torch.utils.data.DataLoader(self.dataset, batch_size, shuffle=shuffle)
        return data_loader
