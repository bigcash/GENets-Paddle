import os, PIL
import cv2
import numpy as np
from paddle.vision.datasets import ImageFolder
from paddle.vision import transforms
from paddle.io import DataLoader, Dataset


class ImagenetDataset(Dataset):
    def __init__(self, image_root, config, transform=None):
        super().__init__()
        self.images = []
        self.labels = []
        self.names = []
        self.transform = transform
        with open(config, mode="r") as f:
            content = f.read()
            lines = content.split("\n")
            for line in lines:
                if len(line)>0:
                    cols = line.split()
                    self.names.append(cols[0])
                    self.images.append(os.path.join(image_root, cols[0]))
                    self.labels.append(int(cols[1]))

    def __getitem__(self, item):
        # print(self.names[item], self.labels[item])
        img_path = self.images[item]
        label = self.labels[item]
        img = PIL.Image.open(img_path).convert('RGB')
        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        
        # img = img.astype(np.float32)
        # print("houxm", type(img), img.shape, img.dtype)
        if self.transform is not None:
            # print("trans", self.transform)
            img = self.transform(img)
        else:
            raise RuntimeError("transform can not be None!")
        # print("load data suc")
        return img, label

    def __len__(self):
        return len(self.labels)
