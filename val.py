import os, sys, argparse, math, PIL, time
import paddle
from paddle.vision.datasets import ImageFolder, DatasetFolder
from paddle.vision import transforms
from paddle.io import DataLoader

from GENets import genet_large, genet_normal, genet_small
from imagenet_dataset import ImagenetDataset


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size for evaluation.')
    parser.add_argument('--workers', type=int, default=1,
                        help='number of workers to load dataset.')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID.')
    parser.add_argument('--data', type=str, default="/home/lingbao/data/imagenet/imgs",
                        help='ImageNet data directory.')
    parser.add_argument('--config', type=str, default="val_list.txt")
    parser.add_argument('--arch', type=str, default="GENet_large",
                        help='model to be evaluated. Could be GENet_large, GENet_normal, GENet_small')
    parser.add_argument('--params_path', type=str, default='genet_large.pdparams',
                        help='Where to find GENet pretrained parameters.')

    opt, _ = parser.parse_known_args(sys.argv)

    print('Warning!!! The GENets are trained by NVIDIA Apex, should run on amp mode. Otherwise the model accuracy might be harmed.')
    input_image_size = 192
    if opt.arch == 'GENet_large':
        input_image_size = 256
        model = genet_large(pretrained=True, model_path=opt.params_path)
    if opt.arch == 'GENet_normal':
        input_image_size = 192
        model = genet_normal(pretrained=True, model_path=opt.params_path)
    if opt.arch == 'GENet_small':
        input_image_size = 192
        model = genet_small(pretrained=True, model_path=opt.params_path)
    # model = paddle.amp.decorate(models=model, level='O2')

    print('Evaluate {} at {}x{} resolution.'.format(opt.arch, input_image_size, input_image_size))

    if opt.gpu==-1:
        paddle.set_device('cpu')
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
    model.eval()

    acc1_sum = 0
    acc5_sum = 0
    n = 0
    timer_start = time.time()
    device = 'cuda:{}'.format(opt.gpu)
    with paddle.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if opt.gpu == -1:
                output = model(input)
            else:
                input = input.astype('float16')
                with paddle.amp.auto_cast(level='O1'):
                    output = model(input)
            target = target.reshape([-1, 1])
            acc1 = paddle.metric.accuracy(output, target, k=1)
            acc5 = paddle.metric.accuracy(output, target, k=5)
            
            acc1_sum += acc1[0] * input.shape[0]
            acc5_sum += acc5[0] * input.shape[0]
            n += input.shape[0]

            if i % 100 == 0:
                print('mini_batch {}, top-1 acc={:4g}%, top-5 acc={:4g}%, number of evaluated images={}'.format(i, acc1[0].item()*100, acc5[0].item()*100, n))
            pass
        pass
    pass

    acc1_avg = acc1_sum / n
    acc5_avg = acc5_sum / n

    timer_end = time.time()
    speed = float(n) / (timer_end - timer_start)

    print('*** arch={}, validation top-1 acc={}%, top-5 acc={}%, number of evaluated images={}, speed={:4g} img/s'.format(
          opt.arch, acc1_avg.item()*100, acc5_avg.item()*100, n, speed))
