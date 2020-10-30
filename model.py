import torch as t
from torch import nn
from torch.nn import functional as F


class HSBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, s, w):
        super(HSBlock, self).__init__()
        self.s = s
        self.w = w
        self.first_pointwise = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=self.s * self.w, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=self.s * self.w),
            nn.ReLU()
        )
        self.split_convs = nn.Sequential()
        split_in_channels = []
        m = 0
        last_pointwise_in_channels = 0
        for i in range(s - 1):
            m = self.w + m // 2
            if i != s - 2:
                last_pointwise_in_channels += m - m // 2
            else:
                last_pointwise_in_channels += m
            split_in_channels.append(m)
            self.split_convs.add_module("conv_%d" % (i,), nn.Sequential(
                nn.Conv2d(in_channels=m, out_channels=m, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_features=m),
                nn.ReLU()
            ))
        last_pointwise_in_channels += self.w
        self.last_pointwise = nn.Sequential(
            nn.Conv2d(in_channels=last_pointwise_in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_channels)
        )

    def forward(self, x):
        output = []
        x = self.first_pointwise(x)
        convs = list(self.split_convs.children())
        next_input_part = None
        for i in range(self.s):
            start_index = i * self.w
            end_index = (i + 1) * self.w
            if i == 0:
                output.append(x[:, start_index:end_index, :, :])
                continue
            m = convs[i - 1]
            if i == 1:
                current_m_output = m(x[:, start_index:end_index, :, :])
            else:
                current_m_output = m(t.cat((x[:, start_index:end_index, :, :], next_input_part), dim=1))
            if i != self.s - 1:
                output_channel_count = current_m_output.size()[1] - current_m_output.size()[1] // 2
                output_start_index = current_m_output.size()[1] - output_channel_count
            else:
                output_start_index = 0
            output_part = current_m_output[:, output_start_index:, :, :]
            next_input_part = current_m_output[:, :output_start_index, :, :]
            output.append(output_part)
        output = t.cat(tuple(output), dim=1)
        output = self.last_pointwise(output)
        return output


class Conv1X1(nn.Module):

    def __init__(self, in_channels, out_channels, is_nonlinear, stride=1):
        super(Conv1X1, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_channels)
        )
        if is_nonlinear:
            self.block.add_module("relu", nn.ReLU())

    def forward(self, x):
        return self.block(x)


class Conv3X3(nn.Module):

    def __init__(self, in_channels, out_channels, stride, is_nonlinear, is_hs_resnet, hs_s, hs_w):
        super(Conv3X3, self).__init__()
        if not is_hs_resnet:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(num_features=out_channels)
            )
        else:
            self.block = nn.Sequential(
                HSBlock(in_channels=in_channels, out_channels=out_channels, stride=stride, s=hs_s, w=hs_w),
                nn.BatchNorm2d(num_features=out_channels)
            )
        if is_nonlinear:
            self.block.add_module("relu", nn.ReLU())

    def forward(self, x):
        return self.block(x)


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, is_hs_resnet, hs_s, hs_w):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            Conv3X3(in_channels=in_channels, out_channels=out_channels, stride=stride, is_nonlinear=True, is_hs_resnet=is_hs_resnet, hs_s=hs_s, hs_w=hs_w),
            Conv3X3(in_channels=out_channels, out_channels=out_channels, stride=1, is_nonlinear=False, is_hs_resnet=is_hs_resnet, hs_s=hs_s, hs_w=hs_w)
        )
        if stride == 2 or in_channels != out_channels:
            self.downsample = Conv1X1(in_channels=in_channels, out_channels=out_channels, stride=stride, is_nonlinear=True)

    def forward(self, x):
        orig = x
        output = self.block(x)
        if self.stride == 2 or self.in_channels != self.out_channels:
            orig = self.downsample(x)
        output = output + orig
        output = F.relu(output)
        return output


class Bottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, stride, is_hs_resnet, hs_s, hs_w):
        super(Bottleneck, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        middle_channels = int(self.out_channels / 4)
        self.block = nn.Sequential(
            Conv1X1(in_channels=in_channels, out_channels=middle_channels, is_nonlinear=True),
            Conv3X3(in_channels=middle_channels, out_channels=middle_channels, stride=stride, is_nonlinear=True, is_hs_resnet=is_hs_resnet, hs_s=hs_s, hs_w=hs_w),
            Conv1X1(in_channels=middle_channels, out_channels=out_channels, is_nonlinear=False)
        )
        if self.stride == 2 or in_channels != out_channels:
            self.downsample = Conv1X1(in_channels=in_channels, out_channels=out_channels, stride=stride, is_nonlinear=True)

    def forward(self, x):
        orig = x
        output = self.block(x)
        if self.stride == 2 or self.in_channels != self.out_channels:
            orig = self.downsample(x)
        output = orig + output
        output = F.relu(output)
        return output


class ResNet(nn.Module):

    def __init__(self, resnet_name, in_channels, block, layers, num_classes, is_hs_resnet, hs_s, hs_w):
        super(ResNet, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv = nn.Sequential()
        for i, layer in enumerate(layers):
            current_layer = nn.Sequential()
            if i == 0:
                in_channels = 64 * 2 ** i
            else:
                in_channels = out_channels
            if resnet_name in ["resnet50", "resnet101", "resnet152"]:
                out_channels = 64 * 2 ** i * 4
            else:
                out_channels = 64 * 2 ** i
            for l in range(layer):
                if l > 0:
                    in_channels = out_channels
                if i > 0 and l == 0:
                    stride = 2
                else:
                    stride = 1
                current_layer.add_module("%d_%d" % (i, l), block(in_channels=in_channels, out_channels=out_channels, stride=stride, is_hs_resnet=is_hs_resnet, hs_s=hs_s, hs_w=hs_w))
            self.conv.add_module("layer%d" % (i,), current_layer)
        self.avg = nn.AdaptiveAvgPool2d(output_size=1)
        if resnet_name in ["resnet50", "resnet101", "resnet152"]:
            in_features = 2048
        else:
            in_features = 512
        self.cls = nn.Linear(in_features=in_features, out_features=num_classes)

    def forward(self, x):
        output = self.head(x)
        output = self.conv(output)
        output = self.avg(output)
        output = output.view((output.size()[0], -1))
        output = self.cls(output)
        return output


def resnet18(in_channels, num_classes):
    """

    :param in_channels: channels of input data
    :param num_classes: category number
    :return:
    """
    model = ResNet(resnet_name="resnet18", in_channels=in_channels, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes, is_hs_resnet=False, hs_s=5, hs_w=15)
    return model


def resnet34(in_channels, num_classes):
    """

        :param in_channels: channels of input data
        :param num_classes: category number
        :return:
    """
    model = ResNet(resnet_name="resnet34", in_channels=in_channels, block=BasicBlock, layers=[3, 4, 6, 3], num_classes=num_classes, is_hs_resnet=False, hs_s=5, hs_w=15)
    return model


def resnet50(in_channels, num_classes):
    """

        :param in_channels: channels of input data
        :param num_classes: category number
        :return:
    """
    model = ResNet(resnet_name="resnet50", in_channels=in_channels, block=Bottleneck, layers=[3, 4, 6, 3], num_classes=num_classes, is_hs_resnet=False, hs_s=5, hs_w=15)
    return model


def resnet101(in_channels, num_classes):
    """

        :param in_channels: channels of input data
        :param num_classes: category number
        :return:
    """
    model = ResNet(resnet_name="resnet101", in_channels=in_channels, block=Bottleneck, layers=[3, 4, 23, 2], num_classes=num_classes, is_hs_resnet=False, hs_s=5, hs_w=15)
    return model


def resnet152(in_channels, num_classes):
    """

        :param in_channels: channels of input data
        :param num_classes: category number
        :return:
    """
    model = ResNet(resnet_name="resnet152", in_channels=in_channels, block=Bottleneck, layers=[3, 8, 36, 3], num_classes=num_classes, is_hs_resnet=False, hs_s=5, hs_w=15)
    return model


def hs_resnet18(in_channels, num_classes, s=5, w=64):
    """

    :param in_channels: channels of input data
    :param num_classes: category number
    :param s: hsresnet parameter s, split branch count
    :param w: hsresnet parameter w, channels of every splited brach
    :return:
    """
    model = ResNet(resnet_name="resnet18", in_channels=in_channels, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes, is_hs_resnet=True, hs_s=s, hs_w=w)
    return model


def hs_resnet34(in_channels, num_classes, s=5, w=64):
    """

        :param in_channels: channels of input data
        :param num_classes: category number
        :param s: hsresnet parameter s, split branch count
        :param w: hsresnet parameter w, channels of every splited brach
        :return:
    """
    model = ResNet(resnet_name="resnet34", in_channels=in_channels, block=BasicBlock, layers=[3, 4, 6, 3], num_classes=num_classes, is_hs_resnet=True, hs_s=s, hs_w=w)
    return model


def hs_resnet50(in_channels, num_classes, s=5, w=64):
    """

        :param in_channels: channels of input data
        :param num_classes: category number
        :param s: hsresnet parameter s, split branch count
        :param w: hsresnet parameter w, channels of every splited brach
        :return:
    """
    model = ResNet(resnet_name="resnet50", in_channels=in_channels, block=Bottleneck, layers=[3, 4, 6, 3], num_classes=num_classes, is_hs_resnet=True, hs_s=s, hs_w=w)
    return model


def hs_resnet101(in_channels, num_classes, s=5, w=64):
    """

        :param in_channels: channels of input data
        :param num_classes: category number
        :param s: hsresnet parameter s, split branch count
        :param w: hsresnet parameter w, channels of every splited brach
        :return:
    """
    model = ResNet(resnet_name="resnet101", in_channels=in_channels, block=Bottleneck, layers=[3, 4, 23, 2], num_classes=num_classes, is_hs_resnet=True, hs_s=s, hs_w=w)
    return model


def hs_resnet152(in_channels, num_classes, s=5, w=64):
    """

        :param in_channels: channels of input data
        :param num_classes: category number
        :param s: hsresnet parameter s, split branch count
        :param w: hsresnet parameter w, channels of every splited brach
        :return:
    """
    model = ResNet(resnet_name="resnet152", in_channels=in_channels, block=Bottleneck, layers=[3, 8, 36, 3], num_classes=num_classes, is_hs_resnet=True, hs_s=s, hs_w=w)
    return model


if __name__ == "__main__":
    model = hs_resnet101(in_channels=3, num_classes=1000, s=5, w=64).cuda(0)
    t.save(model.state_dict(), "model.pth")
    d = t.randn(2, 3, 256, 256).cuda(0)
    output = model(d)
    print(output.size())






