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


class MNIST:
    DIMENSIONALITY = 28
    N_CHANNELS = 1
    N_CLASSES = 10

    NORMALIZATION = ((0.1307,), (0.3081,))

    DEFAULT_MNIST_TRAIN_TRANSFORMS = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(*NORMALIZATION),
        ]
    )

    @staticmethod
    def get_data(train: bool, data_path: str = './data/'):
        return datasets.MNIST(
            root=data_path,
            train=train,
            download=True,
            transform=MNIST.DEFAULT_MNIST_TRAIN_TRANSFORMS,
        )


class DatasetGeneratorPyTorch:
    DATASETS = {
        "CIFAR10": CIFAR10,
        'MNIST': MNIST
    }

    def __init__(self, dataset_name: str, data_path: str = './data/'):
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.train_dataset = None
        self.test_dataset = None
        self.dimensionality = None
        self.n_channels = None
        self.n_classes = None

    def init_dataset(self):
        self.train_dataset = self.DATASETS[self.dataset_name].get_data(train=True, data_path=self.data_path)
        self.test_dataset = self.DATASETS[self.dataset_name].get_data(train=False, data_path=self.data_path)
        self.dimensionality = self.DATASETS[self.dataset_name].DIMENSIONALITY
        self.n_channels = self.DATASETS[self.dataset_name].N_CHANNELS
        self.n_classes = self.DATASETS[self.dataset_name].N_CLASSES
        return self.train_dataset, self.test_dataset

    def get_data_loader(self, batch_size: int, shuffle=True, train: bool = True):
        if train:
            return torch.utils.data.DataLoader(self.train_dataset, batch_size, shuffle=shuffle)
        else:
            return torch.utils.data.DataLoader(self.test_dataset, batch_size, shuffle=shuffle)

    def get_random_sampled_data_loader(self, sample_percentage: float, batch_size: int, train: bool = True):
        if train:
            dataset = self.train_dataset
        else:
            dataset = self.test_dataset
        subset_indices = self._get_random_indices_of_data(data=dataset, sample_percentage=sample_percentage)
        random_subset_data = self._get_subset_of_data(data=dataset, subset_indices=subset_indices)
        return torch.utils.data.DataLoader(random_subset_data, batch_size, shuffle=False), subset_indices

    def _get_random_indices_of_data(self, data, sample_percentage):
        data_len = len(data)
        data_indices_permuted = torch.randperm(data_len)
        num_samples = int(sample_percentage * data_len)
        subset_indices = data_indices_permuted[:num_samples]
        return subset_indices

    def _get_subset_of_data(self, data, subset_indices):
        return torch.utils.data.Subset(dataset=data, indices=subset_indices)
