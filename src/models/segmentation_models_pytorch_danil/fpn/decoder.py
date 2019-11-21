import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base.model import Model


class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):

        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3),
                              stride=1, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return x


class FPNBlockAttG(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x, g = x
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x1 * psi + x


class FPNBlock(nn.Module):
    def __init__(self, pyramid_channels, skip_channels):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)

    def forward(self, x):
        x, skip = x

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        skip = self.skip_conv(skip)

        x = x + skip
        return x


class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()

        blocks = [
            Conv3x3GNReLU(in_channels, out_channels, upsample=bool(n_upsamples))
        ]

        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(Conv3x3GNReLU(out_channels, out_channels, upsample=True))

        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)


class SupervisionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, final_channels):
        super().__init__()

        blocks = [Conv3x3GNReLU(in_channels, out_channels, upsample=True),
                  nn.Conv2d(out_channels, final_channels, kernel_size=1, padding=0)]
        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)


class FPNDecoder(Model):

    def __init__(
            self,
            encoder_channels,
            pyramid_channels=256,
            segmentation_channels=128,
            final_upsampling=4,
            final_channels=1,
            hypercolumn=False,
            supervision=False,
            attentionGate=False,
            dropout=0.2,
    ):
        super().__init__()
        # print(encoder_channels)
        # (576, 200, 72, 40, 56)
        self.final_upsampling = final_upsampling
        self.hypercolumn = hypercolumn
        self.supervision = supervision

        if attentionGate:
            print("Use attention")
            self.conv1 = nn.Conv2d(encoder_channels[0], pyramid_channels, kernel_size=(1, 1))
            self.p4 = FPNBlockAttG(encoder_channels[1], pyramid_channels, pyramid_channels)
            self.p3 = FPNBlockAttG(encoder_channels[2], pyramid_channels, pyramid_channels)
            self.p2 = FPNBlockAttG(encoder_channels[3], pyramid_channels, pyramid_channels)

        else:
            self.conv1 = nn.Conv2d(encoder_channels[0], pyramid_channels, kernel_size=(1, 1))
            self.p4 = FPNBlock(pyramid_channels, encoder_channels[1])
            self.p3 = FPNBlock(pyramid_channels, encoder_channels[2])
            self.p2 = FPNBlock(pyramid_channels, encoder_channels[3])

        self.s5 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=3)
        self.s4 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=2)
        self.s3 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=1)
        self.s2 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=0)

        self.dropout = nn.Dropout2d(p=dropout, inplace=True)
        last_conv_channels = segmentation_channels*4 if self.hypercolumn else segmentation_channels
        self.final_conv = nn.Conv2d(last_conv_channels, final_channels, kernel_size=1, padding=0)

        self.initialize()

    def forward(self, x):
        c5, c4, c3, c2, _ = x

        p5 = self.conv1(c5)
        p4 = self.p4([p5, c4])
        p3 = self.p3([p4, c3])
        p2 = self.p2([p3, c2])

        s5 = self.s5(p5)
        s4 = self.s4(p4)
        s3 = self.s3(p3)
        s2 = self.s2(p2)

        if self.hypercolumn:
            x = torch.cat([s5, s4, s3, s2], dim=1)
        else:
            x = s5 + s4 + s3 + s2

        x = self.dropout(x)
        x = self.final_conv(x)

        if self.final_upsampling is not None and self.final_upsampling > 1:
            x = F.interpolate(x, scale_factor=self.final_upsampling, mode='bilinear', align_corners=True)

        return x
