import torch
import paddle
import numpy as np
from reprod_log import ReprodLogger
from reprod_log import ReprodDiffHelper

from utilities import build_paddle_data_pipeline, build_torch_data_pipeline
from utilities import evaluate
from utilities import accuracy_torch
from tools.GENets import genet_large
import GENet


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
    torch_model.half()
    torch_model.to(torch_device)

    accuracy_paddle = paddle.metric.accuracy
    # prepare logger & load data
    reprod_logger = ReprodLogger()
    paddle_dataset, paddle_dataloader = build_paddle_data_pipeline()
    torch_dataset, torch_dataloader = build_torch_data_pipeline()
    for idx, (paddle_batch, torch_batch
              ) in enumerate(zip(paddle_dataloader, torch_dataloader)):
        if idx > 0:
            break
        evaluate(paddle_batch[0], paddle_batch[1], paddle_model,
                 accuracy_paddle, 'paddle', reprod_logger, device)
        evaluate(torch_batch[0], torch_batch[1], torch_model, accuracy_torch,
                 'ref', reprod_logger, torch_device)


if __name__ == "__main__":
    test_forward()

    # load data
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info("./test_tipc/result/metric_ref.npy")
    paddle_info = diff_helper.load_info("./test_tipc/result/metric_paddle.npy")
    print(torch_info, paddle_info)

    # compare result and produce log
    diff_helper.compare_info(torch_info, paddle_info)
    diff_helper.report(path="./test_tipc/result/log/metric_diff.log")
