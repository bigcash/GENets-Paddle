# GENets-Paddle
paper replay in paddle(Neural Architecture Design for GPU-Efficient Networks)

原作者实验中使用的是从训练集中随机抽取50000张未参与LLR-NAS训练的数据，作为验证集。
python val.py --gpu 0 --batch_size 64 --data /home/lingbao/data/imagenet-mini/imagenet-mini-test/ --arch GENet_small --params_dir ./
Warning!!! The GENets are trained by NVIDIA Apex, it is suggested to turn on --use_apex in the evaluation. Otherwise the model accuracy might be harmed.
Evaluate GENet_small at 192x192 resolution.
Using GPU 0.
mini_batch 0, top-1 acc=   0%, top-5 acc=   0%, number of evaluated images=64
mini_batch 100, top-1 acc=   0%, top-5 acc=   0%, number of evaluated images=6464
*** arch=GENet_small, validation top-1 acc=0.0%, top-5 acc=0.01666666567325592%, number of evaluated images=12000, speed=602.529 img/s
