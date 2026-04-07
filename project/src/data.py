import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np


def get_mnist_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    return train_dataset, test_dataset


def split_dataset(train_dataset, num_clients=5):
    data_len = len(train_dataset)
    indices = np.random.permutation(data_len)
    split_size = data_len // num_clients

    client_subsets = []
    for i in range(num_clients):
        start = i * split_size
        end = (i + 1) * split_size if i != num_clients - 1 else data_len
        subset_indices = indices[start:end]
        client_subsets.append(Subset(train_dataset, subset_indices))

    return client_subsets


def get_client_loader(client_dataset, batch_size=32):
    return DataLoader(client_dataset, batch_size=batch_size, shuffle=True)


def get_test_loader(test_dataset, batch_size=64):
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
