import torch
from torchvision import datasets, transforms


class CIFAR10:
    DIMENSIONALITY = 32
    N_CHANNELS = 3
    N_CLASSES = 10

    CIFAR_NORMALIZATION = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    DEFAULT_CIFAR_TRAIN_TRANSFORMS = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*CIFAR_NORMALIZATION),
        ]
    )

    @staticmethod
    def get_data(train: bool, data_path: str = './data/'):
        return datasets.CIFAR10(
            root=data_path,
            train=train,
            download=True,
            transform=CIFAR10.DEFAULT_CIFAR_TRAIN_TRANSFORMS,
        )


class DatasetGeneratorPyTorch:
    DATASETS = {
        "CIFAR10": CIFAR10
    }

    def __init__(self, dataset_name: str, train: bool, data_path: str = './data/'):
        self.dataset_name = dataset_name
        self.train = train
        self.data_path = data_path
        self.dataset = None
        self.dimensionality = None
        self.n_channels = None
        self.n_classes = None

    def init_dataset(self):
        self.dataset = self.DATASETS[self.dataset_name].get_data(train=self.train, data_path=self.data_path)
        self.dimensionality = self.DATASETS[self.dataset_name].DIMENSIONALITY
        self.n_channels = self.DATASETS[self.dataset_name].N_CHANNELS
        self.n_classes = self.DATASETS[self.dataset_name].N_CLASSES
        return self.dataset

    def get_data_loader(self, batch_size, shuffle=True):
        data_loader = torch.utils.data.DataLoader(self.dataset, batch_size, shuffle=shuffle)
        return data_loader
