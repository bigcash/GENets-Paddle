import torch.utils.data as data
import os, PIL

class DatasetTorch(data.Dataset):
    def __init__(self, image_root, config, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform
        with open(config, mode="r") as f:
            content = f.read()
            lines = content.split("\n")
            for line in lines:
                if len(line)>0:
                    cols = line.split()
                    self.images.append(os.path.join(image_root, cols[0]))
                    self.labels.append(int(cols[1]))
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, item):
        img_path = self.images[item]
        label = self.labels[item]
        img = PIL.Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, label