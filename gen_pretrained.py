import paddle
import torch
from GENets import genet_large, genet_normal, genet_small


def gen_pretrained(pth_file, paddle_model, paddle_name):
    checkpoint = torch.load(pth_file, map_location="cpu")
    torch_state_dict = checkpoint["state_dict"]
    model = paddle_model
    state_dict = model.state_dict()
    for k in state_dict.keys():
        k2 = k
        if "_mean" in k:
            k2 = k.replace("_mean", "running_mean")
        if "_variance" in k:
            k2 = k.replace("_variance", "running_var")
        if "module_list.1." in k:
            k2 = k2.replace("module_list.1.", "module_list.1.netblock.")
        if "module_list.9." in k:
            k2 = k2.replace("module_list.9.", "module_list.9.netblock.")
        v1 = paddle.to_tensor(torch_state_dict[k2].numpy())
        if "fc_linear.weight" in k:
            v1 = paddle.transpose(v1, perm=[1, 0])
        state_dict[k] = v1
    model.set_dict(state_dict)
    model = paddle.amp.decorate(model, level="O1")
    paddle.save(model.state_dict(), "%s.pdparams" % paddle_name)


if __name__=="__main__":
    gen_pretrained("GENet_large.pth", genet_large(), "genet_large")
    gen_pretrained("GENet_normal.pth", genet_normal(), "genet_normal")
    gen_pretrained("GENet_small.pth", genet_small(), "genet_small")


