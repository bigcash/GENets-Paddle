# GENets-Paddle
paper replay in paddle(Neural Architecture Design for GPU-Efficient Networks)
Ming Lin, Hesen Chen, Xiuyu Sun, Qi Qian, Hao Li, Rong Jin. Neural Architecture Design for GPU-Efficient Networks. arXiv:2006.14090 [cs.CV]. June 2020. [[arXiv](https://arxiv.org/abs/2006.14090)]

# read paper
- 目标：设计一套GPU推理高效的且高性能的模型结构。设计一个高效的高精度网络，专门针对现代GPU上的快速推理进行优化
- 设计灵感：This design is inspired by the observation that convolutional kernels in the high-level stages are more likely to have low intrinsic rank and different types of convolutions have different kinds of efficiency on GPU.
- 设计原则：low-level stages使用全卷积(XX-Block)，high-level stages使用dw卷积(DW-Block)和bottleneck卷积(BL-Block)。结构如下：
![网络结构](asset/structure.png)
- 设计过程：根据gpu端高效网络的设计原则，通过LLR-NAS来设计高效网络。

# 原论文实验结果
![performance](asset/performance.png)
![performance table](asset/performance_table.png)

# 复现过程
- 数据：[ILSVRC2012_img_val.tar](https://aistudio.baidu.com/aistudio/datasetdetail/68594)
- 预训练模型：从原repo中下载模型后，执行gen_pretrained.py将pytorch模型转换为paddle模型。[paddle模型](https://pan.baidu.com/s/1u-90N6kehZAyYBvM7QgQ8g)(l5hu)：GENet_large、GENet_normal、GENet_small。(原repo提供的pytorch模型：[GENet_large](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/GENet/GENet_large.pth)、[GENet_normal](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/GENet/GENet_normal.pth)、[GENet_small](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/GENet/GENet_small.pth))
- 评估模型：执行val.py，结果如下：
![paddle模型推理](asset/genet_paddle_experiment.png)

# Acknowledgement
感谢[GPU-Efficient-Networks](https://github.com/idstcv/GPU-Efficient-Networks)
