import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


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
    def load_cifar10(batch_size=64):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_set = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
        test_set = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
        return (
            DataLoader(train_set, batch_size=batch_size, shuffle=True),
            DataLoader(test_set, batch_size=batch_size, shuffle=False),
        )
