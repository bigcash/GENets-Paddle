from paddle.vision.datasets import ImageFolder
from paddle.vision import transforms
from paddle.io import DataLoader

val_dir = "~/data/imagenet-mini/imagenet-mini-test"
transformer = transforms.Compose([transforms.CenterCrop(192), transforms.ToTensor()])
val_dataset = ImageFolder(val_dir, transform=transformer)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=4)
for i, (input, lable) in enumerate(val_loader):
    print(input.shape)