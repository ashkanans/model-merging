import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset


class DatasetLoader:
    @staticmethod
    def load_mnist(batch_size=64):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_set = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        test_set = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
        return (
            DataLoader(train_set, batch_size=batch_size, shuffle=True),
            DataLoader(test_set, batch_size=batch_size, shuffle=False),
        )

    @staticmethod
    def load_cifar10(batch_size=64, downsample=True):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_set = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
        test_set = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

        if downsample:
            # Downsample dataset by 50%
            train_set = Subset(train_set, indices=torch.arange(0, len(train_set) // 2))
            test_set = Subset(test_set, indices=torch.arange(0, len(test_set) // 2))

        return (
            DataLoader(train_set, batch_size=batch_size, shuffle=True),
            DataLoader(test_set, batch_size=batch_size, shuffle=False),
        )

    @staticmethod
    def load_fashion_mnist(batch_size=64):
        """
        Load the Fashion-MNIST dataset with transformations.

        :param batch_size: Batch size for the DataLoader.
        :return: Tuple of DataLoader instances for train and test datasets.
        """
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_set = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
        test_set = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)
        return (
            DataLoader(train_set, batch_size=batch_size, shuffle=True),
            DataLoader(test_set, batch_size=batch_size, shuffle=False),
        )
