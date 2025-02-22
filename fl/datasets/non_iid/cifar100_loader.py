import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR100
import numpy as np

# loads CIFAR100 dataset
def get_non_iid_cifar100_dataloader(num_clients=10, num_classes_per_client=2, batch_size=32):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = CIFAR100(root="./data", train=True, download=True, transform=transform)

    # Group indices by class
    class_indices = [[] for _ in range(10)]
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    # Non-IID Split: Each client gets data from a subset of classes
    clients_data = []
    for _ in range(num_clients):
        client_indices = []
        selected_classes = np.random.choice(range(10), num_classes_per_client, replace=False)
        for cls in selected_classes:
            client_indices += np.random.choice(class_indices[cls], 100, replace=False).tolist()
        clients_data.append(Subset(dataset, client_indices))

    # Create DataLoaders for each client
    client_loaders = [DataLoader(client_data, batch_size=batch_size, shuffle=True) for client_data in clients_data]
    return client_loaders
