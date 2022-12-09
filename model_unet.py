import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
import matplotlib.pyplot as plt


def double_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3),
        nn.ReLU(inplace=True)
    )
    return conv


def crop_img(tensor, target_tensor):
    target_size = target_tensor.size()[2]  # assumed square
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta//2
    return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta]


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv1 = double_conv(3, 64)
        self.down_conv2 = double_conv(64, 128)
        self.down_conv3 = double_conv(128, 256)
        self.down_conv4 = double_conv(256, 512)
        self.down_conv5 = double_conv(512, 1024)

        self.up_trans_1 = nn.ConvTranspose2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=2,
            stride=2
        )
        self.up_trans_2 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=2,
            stride=2
        )
        self.up_trans_3 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=2,
            stride=2
        )
        self.up_trans_4 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=2,
            stride=2
        )

        self.up_conv1 = double_conv(1024, 512)
        self.up_conv2 = double_conv(512, 256)
        self.up_conv3 = double_conv(256, 128)
        self.up_conv4 = double_conv(128, 64)

        self.out = nn.Conv2d(
            in_channels=64,
            out_channels=1,
            kernel_size=1
        )

    # forward
    def forward(self, img):

        # encoder
        x1 = self.down_conv1(img)
        x2 = self.max_pool(x1)

        x3 = self.down_conv2(x2)
        x4 = self.max_pool(x3)

        x5 = self.down_conv3(x4)
        x6 = self.max_pool(x5)

        x7 = self.down_conv4(x6)
        x8 = self.max_pool(x7)

        x9 = self.down_conv5(x8)

        # decoder
        x = self.up_trans_1(x9)
        y = crop_img(x7, x)
        x = self.up_conv1(torch.concat([x, y], 1))

        x = self.up_trans_2(x)
        y = crop_img(x5, x)
        x = self.up_conv2(torch.concat([x, y], 1))

        x = self.up_trans_3(x)
        y = crop_img(x3, x)
        x = self.up_conv3(torch.concat([x, y], 1))

        x = self.up_trans_4(x)
        y = crop_img(x1, x)
        x = self.up_conv4(torch.concat([x, y], 1))

        # out
        x = self.out(x)

        print(x.shape)
        # x = x.detach().numpy()
        # x = np.resize(x, (372, 372))
        # plt.imshow(x)
        # plt.show()


# from dataset import CellDataset


if __name__ == "__main__":
    model = UNet()
    image = torch.rand((1, 3, 572, 572))
    # image_dir = "C:/Users/utsav/Desktop/UNET2/data/test_images"
    # mask_dir = "C:/Users/utsav/Desktop/UNET2/data/test_masks"
    # cd = CellDataset(image_dir, mask_dir)
    # x, y = cd.__getitem__(5)
    model.forward(image)


