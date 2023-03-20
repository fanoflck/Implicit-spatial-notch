import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchsummary import summary
import torchvision.models as models


# KPN基本网路单元
class Basic(nn.Module):

    def __init__(self, in_ch, out_ch, g=16, channel_att=False, spatial_att=False):
        super(Basic, self).__init__()
        self.channel_att = channel_att
        self.spatial_att = spatial_att
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(out_ch),
            nn.ReLU())

        if channel_att:
            self.att_c = nn.Sequential(nn.Conv2d(2 * out_ch, out_ch // g, 1, 1, 0), nn.ReLU(),
                                       nn.Conv2d(out_ch // g, out_ch, 1, 1, 0), nn.Sigmoid())
        if spatial_att:
            self.att_s = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3),
                                       nn.Sigmoid())

    def forward(self, data):
        """
        Forward function.
        :param data:
        :return: tensor
        """
        fm = self.conv1(data)
        if self.channel_att:
            # fm_pool = F.adaptive_avg_pool2d(fm, (1, 1)) + F.adaptive_max_pool2d(fm, (1, 1))
            fm_pool = torch.cat([F.adaptive_avg_pool2d(fm, (1, 1)), F.adaptive_max_pool2d(fm, (1, 1))], dim=1)
            att = self.att_c(fm_pool)
            fm = fm * att
        if self.spatial_att:
            fm_pool = torch.cat([torch.mean(fm, dim=1, keepdim=True), torch.max(fm, dim=1, keepdim=True)[0]], dim=1)
            att = self.att_s(fm_pool)
            fm = fm * att
        return fm


class KPN(nn.Module):

    def __init__(self,
                 color=True,
                 kernel_size=[5],
                 channel_att=False,
                 spatial_att=False,
                 upMode='bilinear',
                 core_bias=False):
        super(KPN, self).__init__()
        self.upMode = upMode
        self.core_bias = core_bias
        self.color_channel = 3 if color else 1
        in_channel = (3 if color else 1)
        out_channel = (3 if color else 1) * (np.sum(np.array(kernel_size)**2))
        if core_bias:
            out_channel += (3 if color else 1)
        # 各个卷积层定义
        # 2~5层都是均值池化+3层卷积
        self.conv1 = Basic(in_channel, 64, channel_att=False, spatial_att=False)
        self.conv2 = Basic(64, 128, channel_att=False, spatial_att=False)
        self.conv3 = Basic(128, 256, channel_att=False, spatial_att=False)
        self.conv4 = Basic(256, 512, channel_att=False, spatial_att=False)
        self.conv5 = Basic(512, 512, channel_att=False, spatial_att=False)
        # 6~8层要先上采样再卷积
        self.conv6 = Basic(512 + 512, 512, channel_att=channel_att, spatial_att=spatial_att)
        self.conv7 = Basic(256 + 512, 256, channel_att=channel_att, spatial_att=spatial_att)
        self.conv8 = Basic(256 + 128, out_channel, channel_att=channel_att, spatial_att=spatial_att)
        self.outc = nn.Conv2d(out_channel, out_channel, 1, 1, 0)

        self.kernel_pred = KernelConv(kernel_size, self.core_bias)

        self.apply(self._init_weights)

    # 静态方法无需实例化
    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)

    # 前向传播函数
    def forward(self, data_with_est, data, white_level=1.0):
        """
        forward and obtain pred image directly
        :param data_with_est: if not blind estimation, it is same as data
        :param data:
        :return: pred_img_i and img_pred
        """
        conv1 = self.conv1(data_with_est)
        conv2 = self.conv2(F.avg_pool2d(conv1, kernel_size=2, stride=2))
        conv3 = self.conv3(F.avg_pool2d(conv2, kernel_size=2, stride=2))
        conv4 = self.conv4(F.avg_pool2d(conv3, kernel_size=2, stride=2))
        conv5 = self.conv5(F.avg_pool2d(conv4, kernel_size=2, stride=2))
        # 开始上采样  同时要进行skip connection
        conv6 = self.conv6(torch.cat([conv4, F.interpolate(conv5, scale_factor=2, mode=self.upMode)], dim=1))
        conv7 = self.conv7(torch.cat([conv3, F.interpolate(conv6, scale_factor=2, mode=self.upMode)], dim=1))
        conv8 = self.conv8(torch.cat([conv2, F.interpolate(conv7, scale_factor=2, mode=self.upMode)], dim=1))
        # return channel K*K*3
        core = self.outc(F.interpolate(conv8, scale_factor=2, mode=self.upMode))

        return self.kernel_pred(data, core, white_level)


class KernelConv(nn.Module):
    """
    the class of computing prediction
    """

    def __init__(self, kernel_size=[5], core_bias=False):
        super(KernelConv, self).__init__()
        self.kernel_size = sorted(kernel_size)
        self.core_bias = core_bias

    def _convert_dict(self, core, batch_size, color, height, width):
        """
        make sure the core to be a dict, generally, only one kind of kernel size is suitable for the func.
        :param core: shape: batch_size*(N*K*K)*height*width
        :return: core_out, a dict
        """
        core_out = {}
        core = core.view(batch_size, -1, color, height, width)
        core_out[self.kernel_size[0]] = core[:, 0:self.kernel_size[0]**2, ...]
        bias = None if not self.core_bias else core[:, -1, ...]
        return core_out, bias

    def forward(self, frames, core, white_level=1.0):
        """
        compute the pred image according to core and frames
        :param frames: [batch_size, 3, height, width]
        :param core: [batch_size, dict(kernel), 3, height, width]
        :return:
        """
        if len(frames.size()) == 4:
            batch_size, color, height, width = frames.size()
        else:
            batch_size, height, width = frames.size()
            color = 1
            frames = frames.view(batch_size, color, height, width)

        core, bias = self._convert_dict(core, batch_size, color, height, width)

        img_stack = []
        kernel = self.kernel_size[::-1]
        K = kernel[0]

        if not img_stack:
            frame_pad = F.pad(frames, [K // 2, K // 2, K // 2, K // 2])
            for i in range(K):
                for j in range(K):
                    img_stack.append(frame_pad[:, :, i:i + height, j:j + width])
            img_stack = torch.stack(img_stack, dim=1)
        pred_img = torch.sum(core[K].mul(img_stack), dim=1, keepdim=False)

        # if bias is permitted
        if self.core_bias:
            if bias is None:
                raise ValueError('The bias should not be None.')
            pred_img += bias
        # print('white_level', white_level.size())
        pred_img = pred_img / white_level
        return pred_img


if __name__ == "__main__":
    from torchvision import transforms, utils
    from PIL import Image
    from utils import gaussian_noise, uniform_noise
    img = Image.open('0000fake.png').convert("RGB")

    img_noise = uniform_noise(img, -20, 20)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])
    img, img_noise = transform(img), transform(img_noise)
    # img, img_noise = img.unsqueeze(dim=0), img_noise.unsqueeze(dim=0)

    print(torch.max(img), torch.min(img))

    kpn = KPN(color=True)
    kpn.load_state_dict(torch.load('checkpoints/checkpoints_for_kpn/try_2/dg_49.pth'))
    pred_img = kpn(img, img_noise)

    print(torch.max(pred_img), torch.min(pred_img))

    # utils.save_image(pred_img, 'filtered.png', normalize=True)
    from utils import saving_image
    saving_image(pred_img, 'fake_filter.png')
    saving_image(img_noise, 'fake_noise.png')
    # utils.save_image(img_noise, 'noise.png', normalize=True)
