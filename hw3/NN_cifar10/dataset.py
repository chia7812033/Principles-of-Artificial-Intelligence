from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor


class CIFAR10Dataset(Dataset):

    def __init__(self, mode='train'):
        if mode == 'train' or mode == 'val':
            dataset = CIFAR10(
                root='./data', train=True, transform=ToTensor())

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
                root='./data', train=False, transform=ToTensor())

        self.input_size = 3 * 32 * 32
        self.output_size = 10

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def get_input_size(self):
        return self.input_size

    def get_output_size(self):
        return self.output_size


def prep_dataloader(mode, batch_size):
    dataset = CIFAR10Dataset(mode=mode)
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode == 'train'), num_workers=2)
    return dataloader
