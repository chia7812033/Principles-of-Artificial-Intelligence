from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms


class CIFAR10Dataset(Dataset):

    def __init__(self, mode='train'):

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        if mode == 'train' or mode == 'val':
            dataset = CIFAR10(
                root='./data', train=True, transform=transform)

            val_size = 5000
            train_size = len(dataset) - val_size

            train_set, val_set = random_split(
                dataset, [train_size, val_size])

            if mode == 'train':
                self.dataset = train_set
            else:
                self.dataset = val_set

        elif mode == 'test':
            self.dataset = CIFAR10(
                root='./data', train=False, transform=transform)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


def prep_dataloader(mode, batch_size):
    dataset = CIFAR10Dataset(mode=mode)
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode == 'train'), num_workers=4, pin_memory=True)
    return dataloader
