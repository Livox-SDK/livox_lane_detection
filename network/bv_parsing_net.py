import torch.nn as nn
import torch
from torch.autograd import Variable


class AsyConv(nn.Module):

    def __init__(self,
                 internal_channels,
                 middle_channels,
                 asy_kernel_size,
                 asy_padding,
                 mode,
                 bias,
                 stride=1):
        super(AsyConv, self).__init__()
        self.asyconv1 = nn.Conv2d(internal_channels,
                                  middle_channels,
                                  kernel_size=(1, asy_kernel_size),
                                  stride=stride,
                                  padding=(0, asy_padding),
                                  bias=bias)

        self.asyconv2 = nn.Conv2d(middle_channels,
                                  internal_channels,
                                  kernel_size=(asy_kernel_size, 1),
                                  stride=stride,
                                  padding=(asy_padding, 0),
                                  bias=bias)

        self.bn2 = nn.BatchNorm2d(internal_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.asyconv1(x)
        out = self.asyconv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        return out


class PoolBranch(nn.Module):

    def __init__(self, avg_or_max, kernel_size, stride=1):
        super(PoolBranch, self).__init__()
        assert avg_or_max in ['max', 'avg'], 'PoolBranch mode should be search or train'
        if avg_or_max == 'avg':
            self.pool = torch.nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=1)
        elif avg_or_max == 'max':
            self.pool = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=1)
        else:
            raise ValueError('Unknown pool {}'.format(avg_or_max))

    def forward(self, x):
        out = self.pool(x)
        return out


class RegularBottleneckOperation(nn.Module):

    def __init__(self,
                 channels,
                 internal_channels,
                 mode,
                 arc_num,
                 dropout_prob,
                 bias):
        super(RegularBottleneckOperation, self).__init__()
        assert mode in ['search', 'train'], 'mode should be search or train'
        assert arc_num in [0,1,2,3,4], 'ops should be 0,1,2,3,4'


        # Main branch - shortcut connection

        # Extension branch - 1x1 convolution, followed by a regular, dilated or
        # asymmetric convolution, followed by another 1x1 convolution, and,
        # finally, a regularizer (spatial dropout). Number of channels is constant.

        # 1x1 projection convolution
        self.conv1 = nn.Conv2d(
                            channels,
                            internal_channels,
                            kernel_size=1,
                            stride=1,
                            bias=bias)
        if mode == 'search':
            self.bn1 = nn.BatchNorm2d(internal_channels, track_running_stats=False )
        else:
            self.bn1 = nn.BatchNorm2d(internal_channels)

        self.relu = nn.ReLU(inplace=True)

        # If the convolution is asymmetric we split the main convolution in
        # two. Eg. for a 5x5 asymmetric convolution we have two convolution:
        # the first is 5x1 and the second is 1x5.
        #middle_channels = int(round(internal_channels / 2,0))
        middle_channels = 8
        if arc_num == 2:
            self.ext_conv2 = AsyConv( internal_channels, 
                                        middle_channels,
                                        asy_kernel_size=3,
                                        asy_padding=1,
                                        bias=bias,
                                        stride=1,
                                        mode = mode )
        if arc_num == 1:
            self.ext_conv2 = AsyConv( internal_channels, 
                                        middle_channels,
                                        asy_kernel_size=5,
                                        asy_padding=2,
                                        bias=bias,
                                        stride=1,
                                        mode = mode ) 
        if arc_num == 0:
            self.ext_conv2 = AsyConv( internal_channels, 
                                        middle_channels,
                                        asy_kernel_size=7,
                                        asy_padding=3,
                                        bias=bias,
                                        stride=1,
                                        mode = mode ) 
        if arc_num == 3:
            self.ext_conv2 = PoolBranch( avg_or_max='avg',
                                            kernel_size=3,
                                            stride=1)
        if arc_num == 4:
            self.ext_conv2 = PoolBranch( avg_or_max='max',
                                            kernel_size=3,
                                            stride=1)

        self.conv3 = nn.Conv2d(internal_channels,
                            channels,
                            kernel_size=1,
                            stride=1,
                            bias=bias)
        if mode == 'search':
            self.bn3 = nn.BatchNorm2d(channels, track_running_stats=False)
        else:
            self.bn3 = nn.BatchNorm2d(channels)

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

    def forward(self, x):
        # Main branch shortcut
        main = x

        # Extension branch
        ext = self.conv1(x)
        ext = self.bn1(ext)
        ext = self.relu(ext)
        ext = self.ext_conv2(ext)
        ext = self.conv3(ext)
        ext = self.bn3(ext)
        ext = self.ext_regul(ext)

        # Add main and extension branches
        out = main + ext
        return out



class RegularBottleneck(nn.Module):

    def __init__(self,
                 channels,
                 internal_channels,
                 mode,
                 kernel_size=3,
                 padding=0,
                 dilation=1,
                 asymmetric=False,
                 dropout_prob=0,
                 bias=False):
        super(RegularBottleneck, self).__init__()
        #assert mode in ['search', 'train'], 'mode should be search or train'
        # Main branch - shortcut connection

        # Extension branch - 1x1 convolution, followed by a regular, dilated or
        # asymmetric convolution, followed by another 1x1 convolution, and,
        # finally, a regularizer (spatial dropout). Number of channels is constant.

        # 1x1 projection convolution
        # self.as = asymmetric
        self.conv1 = nn.Conv2d(
            channels,
            internal_channels,
            kernel_size=1,
            stride=1,
            bias=bias)

        self.bn1 = nn.BatchNorm2d(internal_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.hehe = asymmetric
        # If the convolution is asymmetric we split the main convolution in
        # two. Eg. for a 5x5 asymmetric convolution we have two convolution:
        # the first is 5x1 and the second is 1x5.
        if asymmetric:
            # middle_channels = int(round(internal_channels / 2.0))
            middle_channels = 8
            # self.ext_conv2 = nn.Sequential(
            self.conv2_1 = nn.Conv2d(
                internal_channels,
                middle_channels,
                kernel_size=(kernel_size, 1),
                stride=1,
                padding=(padding, 0),
                dilation=dilation,
                bias=bias)

            self.conv2_2 = nn.Conv2d(
                middle_channels,
                internal_channels,
                kernel_size=(1, kernel_size),
                stride=1,
                padding=(0, padding),
                dilation=dilation,
                bias=bias)

            self.bn2_2 = nn.BatchNorm2d(internal_channels)
            self.relu2_2 = nn.ReLU(inplace=True)

        else:
            self.conv2_3 = nn.Conv2d(
                internal_channels,
                internal_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                dilation=dilation,
                bias=bias)

            self.bn2_3 = nn.BatchNorm2d(internal_channels)
            self.relu2_3 = nn.ReLU(inplace=True)

        # 1x1 expansion convolution
        self.conv3 = nn.Conv2d(
            internal_channels,
            channels,
            kernel_size=1,
            stride=1,
            bias=bias)

        self.bn3 = nn.BatchNorm2d(channels)

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

    def forward(self, x):
        # Main branch shortcut
        main = x

        # Extension branch
        ext = self.conv1(x)
        ext = self.bn1(ext)
        ext = self.relu1(ext)
        if self.hehe:
            ext = self.conv2_1(ext)
            ext = self.conv2_2(ext)
            ext = self.bn2_2(ext)
            ext = self.relu2_2(ext)
        else:
            ext = self.conv2_3(ext)
            ext = self.bn2_3(ext)
            ext = self.relu2_3(ext)
        ext = self.conv3(ext)
        ext = self.bn3(ext)

        ext = self.ext_regul(ext)

        # Add main and extension branches
        out = main + ext
        return out


class InitialBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 mode,
                 kernel_size=2,
                 padding=0,
                 bias=False):
        super(InitialBlock, self).__init__()
        assert mode in ['search', 'train'], 'mode should be search or train'
        self.conv_init = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=padding,
            bias=bias)

        self.bn_init = nn.BatchNorm2d(out_channels)

        self.relu_init = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv_init(x)
        out = self.bn_init(out)
        out = self.relu_init(out)
        return out

class DownsamplingBottleneck(nn.Module):

    def __init__(self,
                 in_channels,
                 internal_channels,
                 out_channels,
                 mode,
                 kernel_size=3,
                 padding=0,
                 dropout_prob=0,
                 bias=False):
        super(DownsamplingBottleneck, self).__init__()
        # Main branch
        self.main_max1 = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=2,
                stride=2),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                bias=bias))

        # Extension branch - 2x2 convolution, followed by a regular, dilated or
        # asymmetric convolution, followed by another 1x1 convolution. Number
        # of channels is doubled.

        # 2x2 projection convolution with stride 2
        # self.ext_conv1 = nn.Sequential(
        self.conv1 = nn.Conv2d(
            in_channels,
            internal_channels,
            kernel_size=2,
            stride=2,
            bias=bias)
        self.bn1 = nn.BatchNorm2d(internal_channels)
        self.relu1 = nn.ReLU(inplace=True)

        # Convolution
        # self.ext_conv2 = nn.Sequential(
        self.conv2 = nn.Conv2d(
            internal_channels,
            internal_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=bias)
        self.bn2 = nn.BatchNorm2d(internal_channels)
        self.relu2 = nn.ReLU(inplace=True)

        # 1x1 expansion convolution
        # self.ext_conv3 = nn.Sequential(
        self.conv3 = nn.Conv2d(
            internal_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            bias=bias)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

    def forward(self, x):
        # Main branch shortcut
        main = self.main_max1(x)

        # Extension branch
        ext = self.conv1(x)
        ext = self.bn1(ext)
        ext = self.relu1(ext)
        ext = self.conv2(ext)
        ext = self.bn2(ext)
        ext = self.relu2(ext)
        ext = self.conv3(ext)
        ext = self.bn3(ext)
        ext = self.ext_regul(ext)

        # Add main and extension branches
        out = main + ext
        return out


class UpsamplingBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 internal_channels,
                 out_channels,
                 mode,
                 bias=False):
        super(UpsamplingBlock, self).__init__()
        assert mode in ['search', 'train'], 'mode should be search or train'
        self.conv1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, bias=bias)

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels, internal_channels, kernel_size=3, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(internal_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(
            internal_channels, out_channels, kernel_size=3, padding=1, bias=bias)
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.upsample(out)
        return out


class BVParsingNet(nn.Module):
    def __init__(self, sample_arc=[0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1], num_classes=33, in_channels=2, mode="train"):
        super(BVParsingNet, self).__init__()
        self.sample_arc = sample_arc
        self.mode = mode
        assert len(self.sample_arc) == 16, 'Cant match length between arc and space'

        self.initial_block = InitialBlock(in_channels=in_channels, out_channels=int(16 / 2), mode=self.mode)#downsampel to 1/2
        # 1
        self.downsample1 = DownsamplingBottleneck( #downsampel to 1/4
            in_channels=int(16 / 2),
            internal_channels=int(16 / 2),
            out_channels=int(64 / 2),
            mode=self.mode,
            kernel_size=3,
            padding=1,
            dropout_prob=0.01,
            bias=False)
        # 2~4
        self.regular2 = RegularBottleneck(
            channels=int(64 / 2),
            internal_channels=int(16 / 2),
            mode=self.mode,
            kernel_size=3,
            padding=1,
            dilation=1,
            asymmetric=True,
            dropout_prob=0.01,
            bias=False)
        self.regular3 = RegularBottleneck(
            channels=int(64 / 2),
            internal_channels=int(16 / 2),
            mode=self.mode,
            kernel_size=3,
            padding=1,
            dilation=1,
            asymmetric=True,
            dropout_prob=0.01,
            bias=False)
        self.regular4 = RegularBottleneck(
            channels=int(64 / 2),
            internal_channels=int(16 / 2),
            mode=self.mode,
            kernel_size=3,
            padding=1,
            dilation=1,
            asymmetric=True,
            dropout_prob=0.01,
            bias=False)
        # 5
        self.regular5 = DownsamplingBottleneck( #downsampel to 1/8
            in_channels=int(64 / 2),
            internal_channels=int(32 / 2),
            mode=self.mode,
            out_channels=int(128 / 2),
            kernel_size=3,
            padding=1,
            dropout_prob=0.1,
            bias=False)
        # 6~21
        self.regular6 = RegularBottleneckOperation(
            channels=int(128 / 2),
            internal_channels=int(64 / 2),
            arc_num=sample_arc[0],
            dropout_prob=0.1,
            mode=self.mode,
            bias=False
        )
        self.regular7 = RegularBottleneckOperation(
            channels=int(128 / 2),
            internal_channels=int(32 / 2),
            arc_num=sample_arc[1],
            dropout_prob=0.1,
            mode=self.mode,
            bias=False
        )
        self.regular8 = RegularBottleneckOperation(
            channels=int(128 / 2),
            internal_channels=int(64 / 2),
            arc_num=sample_arc[2],
            dropout_prob=0.1,
            mode=self.mode,
            bias=False
        )
        self.regular9 = RegularBottleneckOperation(
            channels=int(128 / 2),
            internal_channels=int(32 / 2),
            arc_num=sample_arc[3],
            dropout_prob=0.1,
            mode=self.mode,
            bias=False
        )
        self.regular10 = RegularBottleneckOperation(
            channels=int(128 / 2),
            internal_channels=int(64 / 2),
            arc_num=sample_arc[4],
            dropout_prob=0.1,
            mode=self.mode,
            bias=False
        )
        self.regular11 = RegularBottleneckOperation(
            channels=int(128 / 2),
            internal_channels=int(32 / 2),
            arc_num=sample_arc[5],
            dropout_prob=0.1,
            mode=self.mode,
            bias=False
        )
        self.regular12 = RegularBottleneckOperation(
            channels=int(128 / 2),
            internal_channels=int(64 / 2),
            arc_num=sample_arc[6],
            dropout_prob=0.1,
            mode=self.mode,
            bias=False
        )
        self.regular13 = RegularBottleneckOperation(
            channels=int(128 / 2),
            internal_channels=int(32 / 2),
            arc_num=sample_arc[7],
            dropout_prob=0.1,
            mode=self.mode,
            bias=False
        )
        self.regular14 = RegularBottleneckOperation(
            channels=int(128 / 2),
            internal_channels=int(64 / 2),
            arc_num=sample_arc[8],
            dropout_prob=0.1,
            mode=self.mode,
            bias=False
        )
        self.regular15 = RegularBottleneckOperation(
            channels=int(128 / 2),
            internal_channels=int(32 / 2),
            arc_num=sample_arc[9],
            dropout_prob=0.1,
            mode=self.mode,
            bias=False
        )
        self.regular16 = RegularBottleneckOperation(
            channels=int(128 / 2),
            internal_channels=int(64 / 2),
            arc_num=sample_arc[10],
            dropout_prob=0.1,
            mode=self.mode,
            bias=False
        )
        self.regular17 = RegularBottleneckOperation(
            channels=int(128 / 2),
            internal_channels=int(32 / 2),
            arc_num=sample_arc[11],
            dropout_prob=0.1,
            mode=self.mode,
            bias=False
        )
        self.regular18 = RegularBottleneckOperation(
            channels=int(128 / 2),
            internal_channels=int(64 / 2),
            arc_num=sample_arc[12],
            dropout_prob=0.1,
            mode=self.mode,
            bias=False
        )
        self.regular19 = RegularBottleneckOperation(
            channels=int(128 / 2),
            internal_channels=int(32 / 2),
            arc_num=sample_arc[13],
            dropout_prob=0.1,
            mode=self.mode,
            bias=False
        )
        self.regular20 = RegularBottleneckOperation(
            channels=int(128 / 2),
            internal_channels=int(64 / 2),
            arc_num=sample_arc[14],
            dropout_prob=0.1,
            mode=self.mode,
            bias=False
        )
        self.regular21 = RegularBottleneckOperation(
            channels=int(128 / 2),
            internal_channels=int(32 / 2),
            arc_num=sample_arc[15],
            dropout_prob=0.1,
            mode=self.mode,
            bias=False
        )

        self.regular22 = RegularBottleneck(
            channels=int(128/2),
            internal_channels=int(32/2),
            mode = self.mode,
            kernel_size=7,
            padding=3,
            dilation=1,
            asymmetric=True,
            dropout_prob=0.1,
            bias=False)
        self.regular23 = RegularBottleneck(
            channels=int(128/2),
            internal_channels=int(32/2),
            mode = self.mode,
            kernel_size=7,
            padding=3,
            dilation=1,
            asymmetric=True,
            dropout_prob=0.1,
            bias=False)
        self.regular24 = RegularBottleneck(
            channels=int(128/2),
            internal_channels=int(32/2),
            mode = self.mode,
            kernel_size=7,
            padding=3,
            dilation=1,
            asymmetric=True,
            dropout_prob=0.1,
            bias=False)
        # 25~27
        self.regular25 = RegularBottleneck(
            channels=int(128/2),
            internal_channels=int(32/2),
            mode = self.mode,
            kernel_size=7,
            padding=3,
            dilation=1,
            asymmetric=True,
            dropout_prob=0.1,
            bias=False)
        self.regular26 = RegularBottleneck(
            channels=int(128/2),
            internal_channels=int(32/2),
            mode = self.mode,
            kernel_size=7,
            padding=3,
            dilation=1,
            asymmetric=True,
            dropout_prob=0.1,
            bias=False)
        self.regular27 = RegularBottleneck(
            channels=int(128/2),
            internal_channels=int(32/2),
            mode = self.mode,
            kernel_size=7,
            padding=3,
            dilation=1,
            asymmetric=True,
            dropout_prob=0.1,
            bias=False)


        # predict0
        self.predict0 = UpsamplingBlock( 
                 in_channels=int(128/2),
                 internal_channels=int(64/2),
                 out_channels=num_classes,
                 mode = self.mode,
                 bias=False)
        # predict1
        self.predict1 = UpsamplingBlock( 
                 in_channels=int(128/2),
                 internal_channels=int(64/2),
                 out_channels=num_classes,
                 mode = self.mode,
                 bias=False)
        # predict2
        self.predict2 = UpsamplingBlock( 
                 in_channels=int(128/2),
                 internal_channels=int(64/2),
                 out_channels=num_classes,
                 mode = self.mode,
                 bias=False)

    def forward(self, x):
        # Initial block
        x = self.initial_block(x)
        x = self.downsample1(x)
        # 2~4
        x = self.regular2(x)
        x = self.regular3(x)
        x = self.regular4(x)
        # 5
        x = self.regular5(x)
        # 6~21
        x = self.regular6(x)
        x = self.regular7(x)
        x = self.regular8(x)
        x = self.regular9(x)
        x = self.regular10(x)
        x = self.regular11(x)
        x = self.regular12(x)
        x = self.regular13(x)
        x = self.regular14(x)
        x = self.regular15(x)
        x = self.regular16(x)
        x = self.regular17(x)
        x = self.regular18(x)
        x = self.regular19(x)
        x = self.regular20(x)
        x = self.regular21(x)

        a0 = x.clone()
        # 22~24
        x = self.regular22(x)
        x = self.regular23(x)
        x = self.regular24(x)
        a1 = x.clone()
        # 25~27
        x = self.regular25(x)
        x = self.regular26(x)
        x = self.regular27(x)
        # print('27',x.shape)
        # predict0
        p0 = self.predict0(a0)
        # predict1:
        p1 = self.predict1(a1)
        # predict2
        p_final = self.predict2(x)
        return p0, p1, p_final 
