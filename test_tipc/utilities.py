import numpy as np
import paddle
import torch
import math
import sys, os
sys.path.append(os.path.dirname("."))

from tools.imagenet_dataset import ImagenetDataset
from dataset_torch import DatasetTorch
from torchvision import transforms as torch_transforms
from paddle.vision import transforms as transforms


def gen_fake_data():
    fake_data = np.random.rand(1, 3, 256, 256).astype(np.float32) - 0.5
    fake_label = np.arange(1).astype(np.int64)
    np.save("./test_tipc/fake_data.npy", fake_data)
    np.save("./test_tipc/fake_label.npy", fake_label)


def evaluate(image, labels, model, acc, tag, reprod_logger, device):
    model.eval()

    if tag == 'paddle':
        input = image.astype('float16')
        with paddle.amp.auto_cast(level='O1'):
            output = model(input)
        labels = labels.reshape([-1, 1])
        acc1 = acc(output, labels, k=1)
        acc5 = acc(output, labels, k=5)
        accracy = [acc1*100, acc5*100]
    else:
        input = image.to(device=device, dtype=torch.float16)
        labels = labels.to(device=device)
        model.half()
        output = model(input)
        accracy = acc(output, labels, topk=(1, 5))

    reprod_logger.add("acc_top1", np.array(accracy[0].cpu()))
    reprod_logger.add("acc_top5", np.array(accracy[1].cpu()))

    reprod_logger.save("./test_tipc/result/metric_{}.npy".format(tag))


def build_paddle_data_pipeline():
    # dataset & data_loader
    input_image_size = 256
    input_image_crop = 0.875
    resize_image_size = int(math.ceil(input_image_size / input_image_crop))
    test_transform = transforms.Compose([
        transforms.Resize(resize_image_size),
        transforms.CenterCrop(input_image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    config = "./test_tipc/lite_data.txt"
    dataset_test = ImagenetDataset("/home/lingbao/data/imagenet/imgs", config, transform=test_transform)

    test_sampler = paddle.io.SequenceSampler(dataset_test)

    test_batch_sampler = paddle.io.BatchSampler(
        sampler=test_sampler, batch_size=4)

    data_loader_test = paddle.io.DataLoader(
        dataset_test, batch_sampler=test_batch_sampler, num_workers=0)

    return dataset_test, data_loader_test


def build_torch_data_pipeline():
    input_image_size = 256
    input_image_crop = 0.875
    resize_image_size = int(math.ceil(input_image_size / input_image_crop))
    test_transform = torch_transforms.Compose([
        torch_transforms.Resize(resize_image_size),
        torch_transforms.CenterCrop(input_image_size),
        torch_transforms.ToTensor(),
        torch_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    config = "./test_tipc/lite_data.txt"
    dataset_test = DatasetTorch("/home/lingbao/data/imagenet/imgs", config, transform=test_transform)
    
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=4,
        sampler=test_sampler,
        num_workers=0,
        pin_memory=True)
    return dataset_test, data_loader_test

def accuracy_torch(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
