import paddle
from GENets import genet_large, genet_small, genet_small

if __name__=="__main__":
    model = genet_large(pretrained=True, model_path="genet_large.pdparams")
    model = paddle.amp.decorate(model, level="O2")
    print(model.state_dict()["module_list.0.netblock.weight"].sum())
