from __future__ import absolute_import, division, print_function
from collections import OrderedDict

from layers import *
from timm.models.layers import trunc_normal_


class get_disp(nn.Module):
    def __init__(self, num_in_layers):
        super(get_disp, self).__init__()
        self.conv1 = Conv3x3(num_in_layers, 2)
        self.normolize = nn.BatchNorm2d(2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.normolize(x)
        return 0.3 * self.sigmoid(x)


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=2, use_skips=True):
        super().__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'bilinear'
        self.scales = scales
        # num_ch_enc = [48, 80, 128]
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = (self.num_ch_enc / 2).astype('int')

        # decoder
        self.convs = OrderedDict()
        for i in range(2, -1, -1):  # 2,1,0
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 2 else self.num_ch_dec[i + 1]  # 部分层有跳跃连接，通道数+1
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)  # 卷积通道数减半
            # print(i, num_ch_in, num_ch_out)
            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            # self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], 2)  # 视差层
            self.convs[("dispconv", s)] = get_disp(self.num_ch_dec[s])

        self.decoder = nn.ModuleList(list(self.convs.values()))
        # self.sigmoid = nn.Sigmoid()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, input_features):
        self.outputs = {}
        x = input_features[-1]
        for i in range(2, -1, -1):  # 2,1,0
            x = self.convs[("upconv", i, 0)](x)  # 卷积操作通道数减半c/2
            x = [upsample(x)]  # 上采样

            if self.use_skips and i > 0:
                x += [input_features[i - 1]]  # [48,80,128]
            x = torch.cat(x, 1)  # [B,C1+C2,W,H]
            x = self.convs[("upconv", i, 1)](x)  # 卷积操作通道数[C1+C2,c/2]
            # 视差图生成：卷积，上采样，激活
            if i in self.scales:
                tmp = self.convs[("dispconv", i)](x)
                udist = nn.functional.interpolate(tmp, scale_factor=2, mode='bilinear', align_corners=True)
                self.outputs[("disp", 0, i)] = udist[:, 0, :, :].unsqueeze(dim=1)
                self.outputs[("disp", "s", i)] = udist[:, 1, :, :].unsqueeze(dim=1)

                # a = tmp[:, 0, :, :]
                # b = tmp[:, 1, :, :]
                # fl = upsample(a.unsqueeze(dim=1), mode='bilinear')  # [B,N,W,H]
                # fr = upsample(b.unsqueeze(dim=1), mode='bilinear')
                # self.outputs[("disp", 0, i)] = 0.3 * self.sigmoid(fl)  # 视差范围约束
                # self.outputs[("disp", "s", i)] = 0.3 * self.sigmoid(fr)

        return self.outputs
