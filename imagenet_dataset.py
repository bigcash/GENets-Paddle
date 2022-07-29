from paddle.vision.datasets import ImageFolder
from paddle.vision import transforms
from paddle.io import DataLoader, Dataset


class ImagenetDataset:
    def __init__(self, data_dir, transform=None):
        super().__init__()

    def __getitem__(self, item):
        return

    def __len__(self):
        return