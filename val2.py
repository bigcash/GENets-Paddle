'''
Copyright (C) 2010-2020 Alibaba Group Holding Limited.

Usage (on V100 with 16GB GPU-memory):
python val.py --data ~/data/imagenet --arch GENet_large --params_dir ./GENet_params/ --batch_size 1528
'''
import os, sys, argparse, math, PIL, time
import paddle
from paddle.vision.datasets import ImageFolder, DatasetFolder
from paddle.vision import transforms
from paddle.io import DataLoader

from imagenet_dataset import ImagenetDataset
from pd_model_trace_large.x2paddle_code import PlainNet

imagenet_data_dir = os.path.expanduser('~/data/imagenet/imgs')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size for evaluation.')
    parser.add_argument('--workers', type=int, default=4,
                        help='number of workers to load dataset.')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID.')
    parser.add_argument('--data', type=str, default=imagenet_data_dir,
                        help='ImageNet data directory.')
    parser.add_argument('--config', type=str, default="val_list_genet.txt")

    opt, _ = parser.parse_known_args(sys.argv)

    print('Warning!!! The GENets are trained by NVIDIA Apex, it is suggested to turn on --use_apex in the evaluation. Otherwise the model accuracy might be harmed.')
    input_image_size = 256
    paddle.disable_static()
    params = paddle.load('/home/lingbao/work/code/GENets-Paddle/pd_model_trace_large/model.pdparams')
    model = PlainNet()
    model.set_dict(params)
    model.eval()

    print('Evaluate at {}x{} resolution.'.format(input_image_size, input_image_size))

    # paddle.set_device('cpu')
    # load dataset
    input_image_crop = 0.875
    resize_image_size = int(math.ceil(input_image_size / input_image_crop))
    test_transform = transforms.Compose([
        transforms.Resize(resize_image_size),
        transforms.CenterCrop(input_image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_dataset = ImagenetDataset(opt.data, opt.config, transform=test_transform)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)

    print('Using GPU {}.'.format(opt.gpu))

    acc1_sum = 0
    acc5_sum = 0
    n = 0
    timer_start = time.time()
    device = 'cuda:{}'.format(opt.gpu)
    with paddle.no_grad():
        for i, (input, target) in enumerate(val_loader):
            # print("lingbao:", len(input), input.shape)
            # input = input.to(device=device, non_blocking=True, dtype=paddle.float16)
            # target = target.to(device=device, non_blocking=True, dtype=paddle.float16)
            # print(input)
            input = input.astype('float16')
            with paddle.amp.auto_cast(level='O1'):
                output = model(input)
            # acc1, acc5 = accuracy(output, target, topk=(1, 5))
            target = target.reshape([-1, 1])
            acc1 = paddle.metric.accuracy(output, target, k=1)
            acc5 = paddle.metric.accuracy(output, target, k=5)
            
            acc1_sum += acc1[0] * input.shape[0]
            acc5_sum += acc5[0] * input.shape[0]
            n += input.shape[0]

            if i % 100 == 0:
                print('mini_batch {}, top-1 acc={:4g}%, top-5 acc={:4g}%, number of evaluated images={}'.format(i, acc1[0].item(), acc5[0].item(), n))
            pass
        pass
    pass

    acc1_avg = acc1_sum / n
    acc5_avg = acc5_sum / n

    timer_end = time.time()
    speed = float(n) / (timer_end - timer_start)

    print('*** arch=large, validation top-1 acc={}%, top-5 acc={}%, number of evaluated images={}, speed={:4g} img/s'.format(
        acc1_avg.item(), acc5_avg.item(), n, speed))
