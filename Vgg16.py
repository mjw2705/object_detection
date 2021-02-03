import torch
import torch.nn as nn


# def ConvB(in_c, out_c, n_iteration):
#     convs = []
#     for _ in range(n_iteration):
#         convs += [nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, padding=1)]
#         in_c = out_c
#         convs += [nn.ReLU()]
#     convs += [nn.MaxPool2d(kernel_size=2)]
#
#     return nn.Sequential(*convs)


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, n_iteration):
        super(ConvBlock, self).__init__()

        self.convs = []
        for i in range(n_iteration):
            self.convs.append(nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, padding=1))
            in_c = out_c
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
            x = self.activation(x)
        x = self.pool(x)

        return x


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.block1 = ConvBlock(3, 64, 2)
        self.block2 = ConvBlock(64, 128, 2)
        self.block3 = ConvBlock(128, 256, 3)
        self.block4 = ConvBlock(256, 512, 3)
        self.block5 = ConvBlock(512, 512, 3)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        y0 = x
        x = self.block4(x)
        y1 = x
        x = self.block5(x)
        y2 = x
        # x = x.view(1, -1)

        return y0, y1, y2


def main():
    inputs = torch.randn(1, 3, 416, 416)
    backbone = VGG16()

    result = backbone(inputs)
    # print(result.shape)
    print(result[0].shape)
    print(result[1].shape)
    print(result[2].shape)


if __name__ == '__main__':
    main()