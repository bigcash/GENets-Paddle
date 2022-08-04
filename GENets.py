import paddle
import paddle.nn as nn
# paddle.set_device("cpu")


class Identity(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class ConvKX(nn.Layer):
    def __init__(self, in_channels=None, out_channels=None, kernel_size=None, stride=None, name=None, **kwargs):
        super(ConvKX, self).__init__(**kwargs)
        self.name = name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (self.kernel_size - 1) // 2
        self.netblock = nn.Conv2D(in_channels=self.in_channels, out_channels=self.out_channels,
                                  kernel_size=self.kernel_size, stride=self.stride,
                                  padding=self.padding, bias_attr=False)

    def forward(self, x):
        output = self.netblock(x)
        return output


class SuperResBlock(nn.Layer):
    def __init__(self, block="KXKX", in_channels=0, out_channels=0, kernel_size=3, stride=1, expansion=1.0, sublayers=1,
                 block_name=None, **kwargs):
        super(SuperResBlock, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.expansion = expansion
        self.stride = stride
        self.sublayers = sublayers
        self.block_name = block_name

        self.shortcut_list = nn.LayerList()
        self.conv_list = nn.LayerList()

        for layerID in range(self.sublayers):
            if layerID == 0:
                current_in_channels = self.in_channels
                current_out_channels = self.out_channels
                current_stride = self.stride
                current_kernel_size = self.kernel_size
            else:
                current_in_channels = self.out_channels
                current_out_channels = self.out_channels
                current_stride = 1
                current_kernel_size = self.kernel_size

            current_expansion_channel = int(round(current_out_channels * self.expansion))
            if block == "KXKX":
                the_conv_block = nn.Sequential(
                    nn.Conv2D(current_in_channels, current_expansion_channel, kernel_size=current_kernel_size,
                              stride=current_stride, padding=(current_kernel_size-1)//2, bias_attr=False),
                    nn.BatchNorm2D(current_expansion_channel),
                    nn.ReLU(),
                    nn.Conv2D(current_expansion_channel, current_out_channels, kernel_size=current_kernel_size,
                              stride=1, padding=(current_kernel_size - 1) // 2, bias_attr=False),
                    nn.BatchNorm2D(current_out_channels),
                )
            elif block == "K1KX":  # not used
                the_conv_block = nn.Sequential(
                    nn.Conv2D(current_in_channels, current_expansion_channel, kernel_size=1,
                              stride=1, padding=0, bias_attr=False),
                    nn.BatchNorm2D(current_expansion_channel),
                    nn.ReLU(),
                    nn.Conv2D(current_expansion_channel, current_out_channels, kernel_size=current_kernel_size,
                              stride=current_stride, padding=(current_kernel_size - 1) // 2, bias_attr=False),
                    nn.BatchNorm2D(current_out_channels),
                )
            elif block == "K1DW":  # not used
                the_conv_block = nn.Sequential(
                    nn.Conv2d(current_in_channels, current_out_channels, kernel_size=1,
                              stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(current_expansion_channel),
                    nn.ReLU(),
                    nn.Conv2d(current_out_channels, current_out_channels, kernel_size=current_kernel_size,
                              stride=current_stride, padding=(current_kernel_size - 1) // 2, bias=False,
                              groups=current_out_channels),
                    nn.BatchNorm2d(current_out_channels),
                )
            elif block == "K1KXK1":
                the_conv_block = nn.Sequential(
                    nn.Conv2D(current_in_channels, current_expansion_channel, kernel_size=1,
                              stride=1, padding=0, bias_attr=False),
                    nn.BatchNorm2D(current_expansion_channel),
                    nn.ReLU(),
                    nn.Conv2D(current_expansion_channel, current_expansion_channel, kernel_size=current_kernel_size,
                              stride=current_stride, padding=(current_kernel_size - 1) // 2, bias_attr=False),
                    nn.BatchNorm2D(current_expansion_channel),
                    nn.ReLU(),
                    nn.Conv2D(current_expansion_channel, current_out_channels, kernel_size=1,
                              stride=1, padding=0, bias_attr=False),
                    nn.BatchNorm2D(current_out_channels),
                )
            elif block == "K1DWK1":
                the_conv_block = nn.Sequential(
                    nn.Conv2D(current_in_channels, current_expansion_channel, kernel_size=1,
                              stride=1, padding=0, bias_attr=False),
                    nn.BatchNorm2D(current_expansion_channel),
                    nn.ReLU(),
                    nn.Conv2D(current_expansion_channel, current_expansion_channel, kernel_size=current_kernel_size,
                              stride=current_stride, padding=(current_kernel_size - 1) // 2, bias_attr=False,
                              groups=current_expansion_channel),
                    nn.BatchNorm2D(current_expansion_channel),
                    nn.ReLU(),
                    nn.Conv2D(current_expansion_channel, current_out_channels, kernel_size=1,
                              stride=1, padding=0, bias_attr=False),
                    nn.BatchNorm2D(current_out_channels),
                )
            else:
                raise RuntimeError(
                    'Error block type {}! '.format(block))
            self.conv_list.append(the_conv_block)

            if current_stride == 1 and current_in_channels == current_out_channels:
                shortcut = Identity()
            else:
                shortcut = nn.Sequential(
                    nn.Conv2D(current_in_channels, current_out_channels, kernel_size=1, stride=current_stride, padding=0,
                              bias_attr=False),
                    nn.BatchNorm2D(current_out_channels))
            self.shortcut_list.append(shortcut)
        pass  # end for
        self.relu = nn.ReLU()

    def forward(self, x):
        output = x
        for block, shortcut in zip(self.conv_list, self.shortcut_list):
            conv_output = block(output)
            output = conv_output + shortcut(output)
            output = self.relu(output)
        return output


class PlainNet(nn.Layer):
    def __init__(self, num_classes=None, plainnet_struct=None, **kwargs):
        super(PlainNet, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.module_list = nn.Sequential(*plainnet_struct)
        self.last_channels = plainnet_struct[-4].out_channels
        self.fc_linear = nn.Linear(self.last_channels, self.num_classes, bias_attr=True)

    def forward(self, x):
        output = x
        output = self.module_list(output)
        output = output.flatten(1)
        output = self.fc_linear(output)
        return output


def genet_small(num_classes=1000, pretrained=False, model_path=""):
    plainnet_struct = [
        ConvKX(3, 13, 3, 2, name="ConvKX"),
        nn.BatchNorm2D(13),
        nn.ReLU(),
        SuperResBlock("KXKX", 13, 48, 3, 2, 1.0, 1),
        SuperResBlock("KXKX", 48, 48, 3, 2, 1.0, 3),
        SuperResBlock("K1KXK1", 48, 384, 3, 2, 0.25, 7),
        SuperResBlock("K1DWK1", 384, 560, 3, 2, 3.0, 2),
        SuperResBlock("K1DWK1", 560, 256, 3, 1, 3.0, 1),
        ConvKX(256, 1920, 1, 1),
        nn.BatchNorm2D(1920),
        nn.ReLU(),
        nn.AdaptiveAvgPool2D(1)
    ]
    model = PlainNet(num_classes=num_classes, plainnet_struct=plainnet_struct)
    if pretrained:
        state_dict = paddle.load(model_path)
        model.set_state_dict(state_dict)
    return model


def genet_normal(num_classes=1000, pretrained=False, model_path=""):
    plainnet_struct = [
        ConvKX(3, 32, 3, 2),
        nn.BatchNorm2D(32),
        nn.ReLU(),
        SuperResBlock("KXKX", 32, 128, 3, 2, 1.0, 1),
        SuperResBlock("KXKX", 128, 192, 3, 2, 1.0, 2),
        SuperResBlock("K1KXK1", 192, 640, 3, 2, 0.25, 6),
        SuperResBlock("K1DWK1", 640, 640, 3, 2, 3.0, 4),
        SuperResBlock("K1DWK1", 640, 640, 3, 1, 3.0, 1),
        ConvKX(640, 2560, 1, 1),
        nn.BatchNorm2D(2560),
        nn.ReLU(),
        nn.AdaptiveAvgPool2D(1)
    ]
    model = PlainNet(num_classes=num_classes, plainnet_struct=plainnet_struct)
    if pretrained:
        state_dict = paddle.load(model_path)
        model.set_state_dict(state_dict)
    return model


def genet_large(num_classes=1000, pretrained=False, model_path=""):
    plainnet_struct = [
        ConvKX(3, 32, 3, 2),
        nn.BatchNorm2D(32),
        nn.ReLU(),
        SuperResBlock("KXKX", 32, 128, 3, 2, 1.0, 1),
        SuperResBlock("KXKX", 128, 192, 3, 2, 1.0, 2),
        SuperResBlock("K1KXK1", 192, 640, 3, 2, 0.25, 6),
        SuperResBlock("K1DWK1", 640, 640, 3, 2, 3.0, 5),
        SuperResBlock("K1DWK1", 640, 640, 3, 1, 3.0, 4),
        ConvKX(640, 2560, 1, 1),
        nn.BatchNorm2D(2560),
        nn.ReLU(),
        nn.AdaptiveAvgPool2D(1)
    ]
    model = PlainNet(num_classes=num_classes, plainnet_struct=plainnet_struct)
    if pretrained:
        state_dict = paddle.load(model_path)
        model.set_state_dict(state_dict)
    return model


if __name__ == '__main__':
    print("GENet small test:")
    model = genet_small(pretrained=True, model_path="genet_small.pdparams")
    t = paddle.randn([4, 3, 192, 192])
    output = model(t)
    print(output.shape)
    print(model.state_dict().keys())
    # print("GENet normal test:")
    # model = genet_normal(pretrained=True, model_path="genet_normal.pdparams")
    # t = paddle.randn([4, 3, 192, 192])
    # output = model(t)
    # print(output.shape)
    # print("GENet large test:")
    # model = genet_large(pretrained=True, model_path="genet_large.pdparams")
    # t = paddle.randn([4, 3, 256, 256])
    # output = model(t)
    # print(output.shape)
