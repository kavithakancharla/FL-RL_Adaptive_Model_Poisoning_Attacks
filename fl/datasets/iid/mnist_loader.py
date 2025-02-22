import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

# loads MNIST dataset
def get_iid_mnist_dataloader(num_clients=10, batch_size=32):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = MNIST(root="./data", train=True, download=True, transform=transform)

    # IID Split: Shuffle and evenly distribute data
    client_data_len = len(dataset) // num_clients
    datasets = random_split(dataset, [client_data_len] * num_clients)

    # Create DataLoaders for each client
    client_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True) for ds in datasets]
    return client_loaders