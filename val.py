'''
Copyright (C) 2010-2020 Alibaba Group Holding Limited.

Usage (on V100 with 16GB GPU-memory):
python val.py --data ~/data/imagenet --arch GENet_large --params_dir ./GENet_params/ --batch_size 1528
'''
import os, sys, argparse, math, PIL, time
import paddle
from paddle.vision.datasets import ImageFolder
from paddle.vision import transforms
from paddle.io import DataLoader

from GENets import genet_large, genet_normal, genet_small

imagenet_data_dir = os.path.expanduser('~/data/imagenet-mini/imagenet-mini-test')


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
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


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size for evaluation.')
    parser.add_argument('--workers', type=int, default=4,
                        help='number of workers to load dataset.')
    parser.add_argument('--use_apex', action='store_true',
                        help='Use NVIDIA Apex (float16 precision).')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID.')
    parser.add_argument('--data', type=str, default=imagenet_data_dir,
                        help='ImageNet data directory.')
    parser.add_argument('--arch', type=str, default="GENet_small",
                        help='model to be evaluated. Could be GENet_large, GENet_normal, GENet_small')
    parser.add_argument('--params_dir', type=str, default='./GENet_models',
                        help='Where to find GENet model structure text files and pretrained parameters.')

    opt, _ = parser.parse_known_args(sys.argv)

    if opt.use_apex:
        from apex import amp
    else:
        print('Warning!!! The GENets are trained by NVIDIA Apex, it is suggested to turn on --use_apex in the evaluation. Otherwise the model accuracy might be harmed.')
    # input_image_size = 192
    if opt.arch == 'GENet_large':
        input_image_size = 256
        model = genet_large()
    if opt.arch == 'GENet_normal':
        input_image_size = 192
        model = genet_normal()
    if opt.arch == 'GENet_small':
        input_image_size = 192
        model = genet_small()

    print('Evaluate {} at {}x{} resolution.'.format(opt.arch, input_image_size, input_image_size))

    paddle.set_device("gpu:0")
    # load dataset
    val_dir = os.path.join(opt.data, 'val')
    input_image_crop = 0.875
    resize_image_size = int(math.ceil(input_image_size / input_image_crop))
    transforms_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_list = [transforms.Resize(resize_image_size),
                      transforms.CenterCrop(input_image_size), transforms.ToTensor(), transforms_normalize]
    transformer = transforms.Compose(transform_list)
    val_dataset = ImageFolder(val_dir, transform=transformer)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)


    # model = GENet.fuse_bn(model)
    
    # load model
    if opt.use_apex:
        model = amp.initialize(model, opt_level="O1")
    else:
        # model = model.half()
        pass

    print('Using GPU {}.'.format(opt.gpu))

    model.eval()
    # model.requires_grad_(False)

    acc1_sum = 0
    acc5_sum = 0
    n = 0
    timer_start = time.time()
    device = 'cuda:{}'.format(opt.gpu)
    with paddle.amp.auto_cast(), paddle.no_grad():
        for i, (input, lable) in enumerate(val_loader):
            print("lingbao:", len(input), input.shape)
            # input = input.to(device=device, non_blocking=True, dtype=paddle.float16)
            # target = target.to(device=device, non_blocking=True, dtype=paddle.float16)
            # output = model(input)
            # acc1, acc5 = accuracy(output, target, topk=(1, 5))
            #
            # acc1_sum += acc1[0] * input.shape[0]
            # acc5_sum += acc5[0] * input.shape[0]
            # n += input.shape[0]

            if i % 100 == 0:
                print('mini_batch {}, top-1 acc={:4g}%, top-5 acc={:4g}%, number of evaluated images={}'.format(i, acc1[0], acc5[0], n))
            pass
        pass
    pass

    acc1_avg = acc1_sum / n
    acc5_avg = acc5_sum / n

    timer_end = time.time()
    speed = float(n) / (timer_end - timer_start)

    print('*** arch={}, validation top-1 acc={}%, top-5 acc={}%, number of evaluated images={}, speed={:4g} img/s'.format(
        opt.arch, acc1_avg, acc5_avg, n, speed))



