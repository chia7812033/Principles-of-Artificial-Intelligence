import torch
import numpy as np

config = {
    'n_epochs': 50,
    'batch_size': 2048,
    'optimizer': 'Adam',
    'optim_hparas': {
        'lr': 0.001,
    },
    'early_stop': 5,
}


def some_settings():
    myseed = 42069
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'
