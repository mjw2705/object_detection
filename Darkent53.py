import torch
import torch.nn as nn


def Darkconv(in_c, out_c, kernel_size, stride, padding=None):
    if padding is None:
        padding = kernel_size // 2

    return nn.Sequential(
        nn.Conv2d(in_channels=in_c,
                  out_channels=out_c,
                  kernel_size=kernel_size,
                  stride=stride,
                  padding=padding),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU()
    )


class DarkRes(nn.Module):
    def __init__(self, in_c, out_c1, out_c2):
        super(DarkRes, self).__init__()

        self.block1 = Darkconv(in_c, out_c1, kernel_size=1, stride=1, padding=0)
        self.block2 = Darkconv(out_c1, out_c2, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        x = self.block1(inputs)
        x = self.block2(x)
        x = torch.add(inputs, x)

        return x


def ResBlock(in_c, out_c1, out_c2, block_num):
    layers = []

    for _ in range(block_num):
        layers += [DarkRes(in_c, out_c1, out_c2)]

    return nn.Sequential(*layers)


class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()
        self.conv1 = Darkconv(in_c=3, out_c=32, kernel_size=3, stride=1, padding=1)
        # 1x
        self.conv2 = Darkconv(in_c=32, out_c=64, kernel_size=3, stride=2, padding=1)
        self.block1 = ResBlock(in_c=64, out_c1=32, out_c2=64, block_num=1)
        # 2x
        self.conv3 = Darkconv(in_c=64, out_c=128, kernel_size=3, stride=2, padding=1)
        self.block2 = ResBlock(in_c=128, out_c1=64, out_c2=128, block_num=2)
        # 8x
        self.conv4 = Darkconv(in_c=128, out_c=256, kernel_size=3, stride=2, padding=1)
        self.block3 = ResBlock(in_c=256, out_c1=128, out_c2=256, block_num=8)
        # 8x
        self.conv5 = Darkconv(in_c=256, out_c=512, kernel_size=3, stride=2, padding=1)
        self.block4 = ResBlock(in_c=512, out_c1=256, out_c2=512, block_num=8)
        # 4x
        self.conv6 = Darkconv(in_c=512, out_c=1024, kernel_size=3, stride=2, padding=1)
        self.block5 = ResBlock(in_c=1024, out_c1=512, out_c2=1024, block_num=4)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.block1(x)
        x = self.conv3(x)
        x = self.block2(x)
        x = self.conv4(x)
        x = self.block3(x)
        y0 = x
        x = self.conv5(x)
        x = self.block4(x)
        y1 = x
        x = self.conv6(x)
        x = self.block5(x)
        y2 = x

        return y0, y1, y2


def main():
    inputs = torch.randn(1, 3, 416, 416)
    backbone = Darknet53()

    result = backbone(inputs)
    print(result)

    print(result[0].shape)
    print(result[1].shape)
    print(result[2].shape)


if __name__ == '__main__':
    main()
