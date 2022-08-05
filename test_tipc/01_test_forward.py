import torch
import paddle
import numpy as np
from reprod_log import ReprodDiffHelper
from reprod_log import ReprodLogger
import sys, os
sys.path.append(os.path.dirname(".")) 

from tools.GENets import genet_large
import GENet
import utilities

def test_forward():
    device = "gpu"  # you can also set it as "cpu"
    torch_device = torch.device("cuda:0" if device == "gpu" else "cpu")
    paddle.set_device(device)

    # load paddle model
    paddle_model = genet_large(pretrained=True, model_path="./test_tipc/genet_large.pdparams")
    paddle_model.eval()

    # load torch model
    torch_model = GENet.genet_large(pretrained=True, root="./test_tipc/")
    torch_model.eval()
    # torch_model.half()
    torch_model.to(torch_device)

    # load data
    inputs = np.load("./test_tipc/fake_data.npy")

    # save the paddle output
    reprod_logger = ReprodLogger()
    # with paddle.amp.auto_cast(level='O1'):
    paddle_out = paddle_model(paddle.to_tensor(inputs, dtype="float32"))
    reprod_logger.add("logits", paddle_out.cpu().detach().numpy())
    reprod_logger.save("./test_tipc/result/forward_paddle.npy")

    # save the torch output
    torch_out = torch_model(
        torch.tensor(
            inputs, dtype=torch.float32).to(torch_device))
    reprod_logger.add("logits", torch_out.cpu().detach().numpy())
    reprod_logger.save("./test_tipc/result/forward_ref.npy")


if __name__ == "__main__":
    utilities.gen_fake_data()
    test_forward()

    # load data
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info("./test_tipc/result/forward_ref.npy")
    paddle_info = diff_helper.load_info("./test_tipc/result/forward_paddle.npy")

    # compare result and produce log
    diff_helper.compare_info(torch_info, paddle_info)
    diff_helper.report(
        path="./test_tipc/result/log/forward_diff.log", diff_threshold=1e-5)
