# Linux GPU/CPU 基础训练推理开发文档

## 目录

- [1. 简介](#1-简介)
- [2. 模型复现流程与规范](#2-数据集和复现精度)
    - [2.1 复现流程]()
    - [2.2 核验点]()

### 1. 简介

该系列文档主要介绍飞桨模型基于 Linux GPU/CPU 基础训练推理开发过程，主要包含3个步骤。

步骤一：参考 《模型复现指南》，完成模型的训练与基于训练引擎的预测程序开发。

步骤二：参考《Linux GPU/CPU 模型推理开发文档》，在基于训练引擎预测的基础上，完成基于Paddle Inference的推理程序开发。

步骤三：参考《Linux GPU/CPU 基础训练推理测试开发文档》，完成Linux GPU/CPU 训练、推理测试功能开发。

### 2. 模型复现流程与规范

#### 2.1 复现流程

start → 模型结构对齐 → 准备小数据集，数据对齐 → 评估指标对齐 → 模型权重对齐 → 预测程序开发 → end

#### 2.2 核验点

##### 2.2.1 小数据集

`lite_data.txt`

##### 2.2.2 代码与精度

- 模型前向对齐

```shell
python test_tipc/01_test_forward.py

INFO 2022-08-05 15:50:30,463 utils.py:118] logits: 
[2022/08/05 15:50:30] root INFO: logits: 
INFO 2022-08-05 15:50:30,463 utils.py:124]      mean diff: check passed: True, value: 9.067356643299718e-08
[2022/08/05 15:50:30] root INFO:        mean diff: check passed: True, value: 9.067356643299718e-08
INFO 2022-08-05 15:50:30,464 ReprodDiffHelper.py:64] diff check passed
[2022/08/05 15:50:30] root INFO: diff check passed
```

- 数据加载对齐

```shell
python test_tipc/02_test_data.py

INFO 2022-08-05 15:52:15,021 utils.py:118] length: 
[2022/08/05 15:52:15] root INFO: length: 
INFO 2022-08-05 15:52:15,021 utils.py:124]      mean diff: check passed: True, value: 0.0
[2022/08/05 15:52:15] root INFO:        mean diff: check passed: True, value: 0.0
INFO 2022-08-05 15:52:15,021 utils.py:118] dataloader_0: 
[2022/08/05 15:52:15] root INFO: dataloader_0: 
INFO 2022-08-05 15:52:15,021 utils.py:124]      mean diff: check passed: True, value: 1.1075422889916808e-07
[2022/08/05 15:52:15] root INFO:        mean diff: check passed: True, value: 1.1075422889916808e-07
INFO 2022-08-05 15:52:15,021 utils.py:118] dataloader_1: 
[2022/08/05 15:52:15] root INFO: dataloader_1: 
INFO 2022-08-05 15:52:15,021 utils.py:124]      mean diff: check passed: True, value: 4.9047844896676907e-08
[2022/08/05 15:52:15] root INFO:        mean diff: check passed: True, value: 4.9047844896676907e-08
INFO 2022-08-05 15:52:15,021 ReprodDiffHelper.py:64] diff check passed
[2022/08/05 15:52:15] root INFO: diff check passed
```

- 评估指标对齐

```shell
python test_tipc/03_test_metric.py

{'acc_top1': array([75.], dtype=float32), 'acc_top5': array([100.], dtype=float32)} {'acc_top1': array([75.], dtype=float32), 'acc_top5': array([100.], dtype=float32)}
INFO 2022-08-05 16:10:30,493 utils.py:118] acc_top1: 
[2022/08/05 16:10:30] root INFO: acc_top1: 
INFO 2022-08-05 16:10:30,493 utils.py:124]      mean diff: check passed: True, value: 0.0
[2022/08/05 16:10:30] root INFO:        mean diff: check passed: True, value: 0.0
INFO 2022-08-05 16:10:30,493 utils.py:118] acc_top5: 
[2022/08/05 16:10:30] root INFO: acc_top5: 
INFO 2022-08-05 16:10:30,493 utils.py:124]      mean diff: check passed: True, value: 0.0
[2022/08/05 16:10:30] root INFO:        mean diff: check passed: True, value: 0.0
INFO 2022-08-05 16:10:30,493 ReprodDiffHelper.py:64] diff check passed
[2022/08/05 16:10:30] root INFO: diff check passed
```
