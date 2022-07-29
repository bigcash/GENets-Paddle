import torch
import numpy as np
import GENet
from x2paddle.convert import pytorch2paddle
# 构建输入
input_data = np.random.rand(1, 3, 192, 192).astype("float32")
# 获取PyTorch Module
torch_module = GENet.genet_small(pretrained=True, root="./")
# 设置为eval模式
torch_module.eval()
# 进行转换
pytorch2paddle(torch_module,
               save_dir="pd_model_trace_small",
               jit_type="trace",
               input_examples=[torch.tensor(input_data)])


# 构建输入
input_data = np.random.rand(1, 3, 192, 192).astype("float32")
# 获取PyTorch Module
torch_module = GENet.genet_normal(pretrained=True, root="./")
# 设置为eval模式
torch_module.eval()
# 进行转换
pytorch2paddle(torch_module,
               save_dir="pd_model_trace_normal",
               jit_type="trace",
               input_examples=[torch.tensor(input_data)])


# 构建输入
input_data = np.random.rand(1, 3, 256, 256).astype("float32")
# 获取PyTorch Module
torch_module = GENet.genet_large(pretrained=True, root="./")
# 设置为eval模式
torch_module.eval()
# 进行转换
pytorch2paddle(torch_module,
               save_dir="pd_model_trace_large",
               jit_type="trace",
               input_examples=[torch.tensor(input_data)])

