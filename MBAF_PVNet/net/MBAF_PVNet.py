import torch
import torch.nn as nn
import torch.nn.functional as F

from net.vgg import VGG
import torch
import torch.nn as nn
import torch.nn.functional as F

class RotEquiAttention(nn.Module):
    def __init__(self, in_channels, mid_channels=None, angles=(0, 45, 90, 135), learnable=True):
        super(RotEquiAttention, self).__init__()
        self.in_c = in_channels
        self.mid_c = mid_channels if mid_channels is not None else max(in_channels // 4, 1)

        # ---- Learnable angles (radians) ----
        init_rads = torch.tensor(angles, dtype=torch.float32) * (torch.pi / 180.0)
        if learnable:
            self.angles = nn.Parameter(init_rads, requires_grad=True)
        else:
            self.register_buffer("angles", init_rads)

        # ---- Channel attention ----
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.in_c, self.mid_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.mid_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_c, self.in_c, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        # ---- Spatial attention ----
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def _theta_from_angle(self, ang_rad, B, device, dtype):
        c = torch.cos(ang_rad).to(device=device, dtype=dtype)
        s = torch.sin(ang_rad).to(device=device, dtype=dtype)
        theta = torch.zeros((B, 2, 3), device=device, dtype=dtype)
        theta[:, 0, 0] = c;  theta[:, 0, 1] = -s
        theta[:, 1, 0] = s;  theta[:, 1, 1] =  c
        return theta  # 无平移

    def forward(self, F_in):
        B, C, H, W = F_in.size()
        device, dtype = F_in.device, F_in.dtype
        fused = torch.zeros_like(F_in)

        for ang_rad in self.angles:
            # 旋转
            theta = self._theta_from_angle(ang_rad, B, device, dtype)
            grid  = F.affine_grid(theta, F_in.size(), align_corners=True)
            F_rot = F.grid_sample(F_in, grid, mode='bilinear', align_corners=True)

            # 通道+空间注意力
            ca = self.channel_att(F_rot)
            F_ca = F_rot * ca
            avg_out = torch.mean(F_ca, dim=1, keepdim=True)
            max_out, _ = torch.max(F_ca, dim=1, keepdim=True)
            sa = self.spatial_att(torch.cat([avg_out, max_out], dim=1))
            F_att = F_ca * sa

            # 反旋回原坐标系
            inv_theta = self._theta_from_angle(-ang_rad, B, device, dtype)
            inv_grid  = F.affine_grid(inv_theta, F_att.size(), align_corners=True)
            F_back    = F.grid_sample(F_att, inv_grid, mode='bilinear', align_corners=True)

            fused = fused + F_back

        return fused / float(len(self.angles))

class DLCM(nn.Module):#Directional Local Context Module (DLCM)
    def __init__(self, in_channels, phi='T'):
        super(DLCM, self).__init__()
        Kernel_size = {'T': 5, 'B': 7, 'S': 5, 'L': 7}[phi]
        groups = {'T': in_channels, 'B': in_channels, 'S': in_channels // 8, 'L': in_channels // 8}[phi]
        def get_valid_group(in_channels):
            for g in [32, 16, 8, 4, 2, 1]:
                if in_channels % g == 0:
                    return g
            return 1
        num_groups = get_valid_group(in_channels)
        pad = Kernel_size // 2

        self.conv1_h = nn.Conv1d(in_channels, in_channels, kernel_size=Kernel_size, padding=pad, groups=groups, bias=False)
        self.conv1_w = nn.Conv1d(in_channels, in_channels, kernel_size=Kernel_size, padding=pad, groups=groups, bias=False)
        self.gn = nn.GroupNorm(num_groups, in_channels)

        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        x_h = torch.mean(x, dim=3, keepdim=True).view(b, c, h)
        x_w = torch.mean(x, dim=2, keepdim=True).view(b, c, w)
        x_h = F.silu(self.gn(self.conv1_h(x_h))).view(b, c, h, 1)
        x_w = F.silu(self.gn(self.conv1_w(x_w))).view(b, c, 1, w)
        spatial_att = x_h * x_w
        channel_att = self.channel_att(x)
        out = x * (1 + spatial_att * channel_att)
        return out
class GatedFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GatedFusion, self).__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 2, kernel_size=1)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, feat1, feat2):
        fused = torch.cat([feat1, feat2], dim=1)   # (B, 2C, H, W)
        gate_logits = self.gate_conv(fused)        # (B, 2, H, W)
        gate = torch.softmax(gate_logits, dim=1)   # (B, 2, H, W)
        gate1 = gate[:, 0:1, :, :]                 # (B, 1, H, W)
        gate2 = gate[:, 1:2, :, :]                 # (B, 1, H, W)
        fused_feat = gate1 * feat1 + gate2 * feat2
        return self.out_conv(fused_feat)           # (B, C, H, W)
class CCAM(nn.Module):#Contextual Convolutional Attention Module (CCAM)
    def __init__(self, dim=512, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size

        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=4, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.value_embed = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim)
        )
        factor = 4
        self.attention_embed = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim // factor, 1, bias=False),
            nn.BatchNorm2d(2 * dim // factor),
            nn.ReLU(),
            nn.Conv2d(2 * dim // factor, kernel_size * kernel_size * dim, 1)
        )
    def forward(self, x):
        bs, c, h, w = x.shape
        k1 = self.key_embed(x)
        v = self.value_embed(x).view(bs, c, -1)
        y = torch.cat([k1, x], dim=1)
        att = self.attention_embed(y)
        att = att.reshape(bs, c, self.kernel_size * self.kernel_size, h, w)
        att = att.mean(2).view(bs, c, -1)
        k2 = F.softmax(att, dim=-1) * v
        k2 = k2.view(bs, c, h, w)
        return k1 + k2

class CCAM_DLCM_Gated(nn.Module):#多分支注意力融合（Multi‐Branch Attention Fusion）
    def __init__(self, in_channels, phi='T', kernel_size=3):
        super(CCAM_DLCM_Gated, self).__init__()
        self.dlcm = DLCM(in_channels, phi)
        self.cca = CCAM(in_channels, kernel_size)

        self.gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        dlcm_out = self.dlcm(x)
        cca_out = self.cca(x)
        fusion = torch.cat([dlcm_out, cca_out], dim=1)
        gate = self.gate(fusion)
        out = gate * cca_out + (1 - gate) * dlcm_out
        return out


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

# for conv5
class MBAFM5(nn.Module):
    def __init__(self, cur_channel):
        super(MBAFM5, self).__init__()
        self.relu = nn.ReLU(True)

        self.cur_b1 = BasicConv2d(cur_channel, cur_channel, 3, padding=1, dilation=1)
        self.cur_b2 = BasicConv2d(cur_channel, cur_channel, 3, padding=2, dilation=2)
        self.cur_b3 = BasicConv2d(cur_channel, cur_channel, 3, padding=3, dilation=3)
        self.cur_b4 = BasicConv2d(cur_channel, cur_channel, 3, padding=4, dilation=4)

        self.cur_all = BasicConv2d(4*cur_channel, cur_channel, 3, padding=1)
        self.cca_dlcm = CCAM_DLCM_Gated(cur_channel, phi='T', kernel_size= 3)

        self.downsample2 = nn.MaxPool2d(2, stride=2)
        self.pre_sa = SpatialAttention()

    def forward(self, x_pre, x_cur):
        # current conv
        x_cur_1 = self.cur_b1(x_cur)
        x_cur_2 = self.cur_b2(x_cur)
        x_cur_3 = self.cur_b3(x_cur)
        x_cur_4 = self.cur_b4(x_cur)
        x_cur_all = self.cur_all(torch.cat((x_cur_1, x_cur_2, x_cur_3, x_cur_4), 1))
        cur_all_fused = self.cca_dlcm(x_cur_all)

        x_pre_ds = self.downsample2(x_pre)
        pre_part = x_cur_all * self.pre_sa(x_pre_ds)
        x_LocAndGlo = cur_all_fused + pre_part + x_cur
        return x_LocAndGlo

class MBAFM4(nn.Module):
    def __init__(self, cur_channel):
        super(MBAFM4, self).__init__()
        self.relu = nn.ReLU(True)

        # 当前层多尺度卷积（与原 ACCoM 结构保持一致）
        self.cur_b1 = BasicConv2d(cur_channel, cur_channel, 3, padding=1, dilation=1)
        self.cur_b2 = BasicConv2d(cur_channel, cur_channel, 3, padding=2, dilation=2)
        self.cur_b3 = BasicConv2d(cur_channel, cur_channel, 3, padding=3, dilation=3)
        self.cur_b4 = BasicConv2d(cur_channel, cur_channel, 3, padding=4, dilation=4)
        self.cur_all = BasicConv2d(4 * cur_channel, cur_channel, 3, padding=1)

        self.cca_dlcm = CCAM_DLCM_Gated(cur_channel, phi='T', kernel_size=3)

        self.downsample2 = nn.MaxPool2d(2, stride=2)
        self.pre_sa = SpatialAttention()

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.lat_sa = SpatialAttention()

    def forward(self, x_pre, x_cur, x_lat):
        x1 = self.cur_b1(x_cur)
        x2 = self.cur_b2(x_cur)
        x3 = self.cur_b3(x_cur)
        x4 = self.cur_b4(x_cur)
        x_cur_all = self.cur_all(torch.cat((x1, x2, x3, x4), dim=1))

        cur_all_fused = self.cca_dlcm(x_cur_all)

        x_pre_ds = self.downsample2(x_pre)
        pre_part = x_cur_all * self.pre_sa(x_pre_ds)

        x_lat_us = self.upsample2(x_lat)
        lat_part = x_cur_all * self.lat_sa(x_lat_us)

        x_LocAndGlo = cur_all_fused + pre_part + lat_part + x_cur
        return x_LocAndGlo

class MBAFM3(nn.Module):
    def __init__(self, cur_channel):
        super(MBAFM3, self).__init__()
        self.relu = nn.ReLU(True)
        self.cur_b1 = BasicConv2d(cur_channel, cur_channel, 3, padding=1, dilation=1)
        self.cur_b2 = BasicConv2d(cur_channel, cur_channel, 3, padding=2, dilation=2)
        self.cur_b3 = BasicConv2d(cur_channel, cur_channel, 3, padding=3, dilation=3)
        self.cur_b4 = BasicConv2d(cur_channel, cur_channel, 3, padding=4, dilation=4)
        self.cur_all = BasicConv2d(4 * cur_channel, cur_channel, 3, padding=1)
        self.cca_dlcm = CCAM_DLCM_Gated(cur_channel, phi='T', kernel_size=3)
        self.downsample2 = nn.MaxPool2d(2, stride=2)
        self.pre_sa = SpatialAttention()
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.lat_sa = SpatialAttention()

    def forward(self, x_pre, x_cur, x_lat):
        x1 = self.cur_b1(x_cur)
        x2 = self.cur_b2(x_cur)
        x3 = self.cur_b3(x_cur)
        x4 = self.cur_b4(x_cur)
        x_cur_all = self.cur_all(torch.cat((x1, x2, x3, x4), dim=1))

        cur_all_fused = self.cca_dlcm(x_cur_all)

        x_pre_ds = self.downsample2(x_pre)
        pre_part = x_cur_all * self.pre_sa(x_pre_ds)

        x_lat_us = self.upsample2(x_lat)
        lat_part = x_cur_all * self.lat_sa(x_lat_us)

        x_LocAndGlo = cur_all_fused + pre_part + lat_part + x_cur
        return x_LocAndGlo



class MBAFM2(nn.Module):
    def __init__(self, cur_channel):
        super(MBAFM2, self).__init__()
        self.relu = nn.ReLU(True)
        self.cur_b1 = BasicConv2d(cur_channel, cur_channel, 3, padding=1, dilation=1)
        self.cur_b2 = BasicConv2d(cur_channel, cur_channel, 3, padding=2, dilation=2)
        self.cur_b3 = BasicConv2d(cur_channel, cur_channel, 3, padding=3, dilation=3)
        self.cur_b4 = BasicConv2d(cur_channel, cur_channel, 3, padding=4, dilation=4)
        self.cur_all = BasicConv2d(4 * cur_channel, cur_channel, 3, padding=1)

        self.dlcm = DLCM(cur_channel, phi='T')

        self.downsample2 = nn.MaxPool2d(2, stride=2)
        self.pre_sa = SpatialAttention()
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.lat_sa = SpatialAttention()

    def forward(self, x_pre, x_cur, x_lat):
        x1 = self.cur_b1(x_cur)
        x2 = self.cur_b2(x_cur)
        x3 = self.cur_b3(x_cur)
        x4 = self.cur_b4(x_cur)
        x_cur_all = self.cur_all(torch.cat((x1, x2, x3, x4), dim=1))

        cur_all_fused = self.dlcm(x_cur_all)

        x_pre_ds = self.downsample2(x_pre)
        pre_part = x_cur_all * self.pre_sa(x_pre_ds)

        x_lat_us = self.upsample2(x_lat)
        lat_part = x_cur_all * self.lat_sa(x_lat_us)

        x_LocAndGlo = cur_all_fused + pre_part + lat_part + x_cur
        return x_LocAndGlo

class MBAFM1(nn.Module):
    def __init__(self, cur_channel):
        super(MBAFM1, self).__init__()
        self.relu = nn.ReLU(True)
        self.cur_b1 = BasicConv2d(cur_channel, cur_channel, 3, padding=1, dilation=1)
        self.cur_b2 = BasicConv2d(cur_channel, cur_channel, 3, padding=2, dilation=2)
        self.cur_b3 = BasicConv2d(cur_channel, cur_channel, 3, padding=3, dilation=3)
        self.cur_b4 = BasicConv2d(cur_channel, cur_channel, 3, padding=4, dilation=4)
        self.cur_all = BasicConv2d(4 * cur_channel, cur_channel, 3, padding=1)

        self.dlcm = DLCM(cur_channel, phi='T')

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.lat_sa = SpatialAttention()

    def forward(self, x_cur, x_lat):
        x1 = self.cur_b1(x_cur)
        x2 = self.cur_b2(x_cur)
        x3 = self.cur_b3(x_cur)
        x4 = self.cur_b4(x_cur)
        x_cur_all = self.cur_all(torch.cat((x1, x2, x3, x4), dim=1))

        cur_all_fused = self.dlcm(x_cur_all)

        x_lat_us = self.upsample2(x_lat)
        lat_part = x_cur_all * self.lat_sa(x_lat_us)

        x_LocAndGlo = cur_all_fused + lat_part + x_cur
        return x_LocAndGlo

class BAB_Decoder(nn.Module):
    def __init__(self, channel_1=1024, channel_2=512, channel_3=256, dilation_1=3, dilation_2=2):
        super(BAB_Decoder, self).__init__()

        self.conv1 = BasicConv2d(channel_1, channel_2, 3, padding=1)
        self.conv1_Dila = BasicConv2d(channel_2, channel_2, 3, padding=dilation_1, dilation=dilation_1)
        self.conv2 = BasicConv2d(channel_2, channel_2, 3, padding=1)
        self.conv2_Dila = BasicConv2d(channel_2, channel_2, 3, padding=dilation_2, dilation=dilation_2)
        self.conv3 = BasicConv2d(channel_2, channel_2, 3, padding=1)
        self.conv_fuse = BasicConv2d(channel_2*3, channel_3, 3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1_dila = self.conv1_Dila(x1)

        x2 = self.conv2(x1)
        x2_dila = self.conv2_Dila(x2)

        x3 = self.conv3(x2)

        x_fuse = self.conv_fuse(torch.cat((x1_dila, x2_dila, x3), 1))

        return x_fuse


class decoder(nn.Module):
    def __init__(self, channel=512):
        super(decoder, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.decoder5 = nn.Sequential(
            BAB_Decoder(512, 512, 512, 3, 2),
            nn.Dropout(0.5),
            TransBasicConv2d(512, 512, kernel_size=2, stride=2,
                              padding=0, dilation=1, bias=False)
        )
        self.S5 = nn.Conv2d(512, 1, 3, stride=1, padding=1)

        self.decoder4 = nn.Sequential(
            BAB_Decoder(1024, 512, 256, 3, 2),
            nn.Dropout(0.5),
            TransBasicConv2d(256, 256, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        )
        self.S4 = nn.Conv2d(256, 1, 3, stride=1, padding=1)

        self.decoder3 = nn.Sequential(
            BAB_Decoder(512, 256, 128, 5, 3),
            nn.Dropout(0.5),
            TransBasicConv2d(128, 128, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        )
        self.S3 = nn.Conv2d(128, 1, 3, stride=1, padding=1)

        self.decoder2 = nn.Sequential(
            BAB_Decoder(256, 128, 64, 5, 3),
            nn.Dropout(0.5),
            TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        )
        self.S2 = nn.Conv2d(64, 1, 3, stride=1, padding=1)

        self.decoder1 = nn.Sequential(
            BAB_Decoder(128, 64, 32, 5, 3)
        )
        self.S1 = nn.Conv2d(32, 1, 3, stride=1, padding=1)


    def forward(self, x5, x4, x3, x2, x1):
        # x5: 1/16, 512; x4: 1/8, 512; x3: 1/4, 256; x2: 1/2, 128; x1: 1/1, 64
        x5_up = self.decoder5(x5)
        s5 = self.S5(x5_up)

        x4_up = self.decoder4(torch.cat((x4, x5_up), 1))
        s4 = self.S4(x4_up)

        x3_up = self.decoder3(torch.cat((x3, x4_up), 1))
        s3 = self.S3(x3_up)

        x2_up = self.decoder2(torch.cat((x2, x3_up), 1))
        s2 = self.S2(x2_up)

        x1_up = self.decoder1(torch.cat((x1, x2_up), 1))
        s1 = self.S1(x1_up)

        return s1, s2, s3, s4, s5


class MBAF_PVNet_VGG(nn.Module):
    def __init__(self, channel=32):
        super(MBAF_PVNet_VGG, self).__init__()
        #Backbone model
        self.vgg = VGG('rgb')

        self.MBAFM5 = MBAFM5(512)
        self.MBAFM4 = MBAFM4(512)
        self.MBAFM3 = MBAFM3(256)
        self.MBAFM2 = MBAFM2(128)
        self.MBAFM1 = MBAFM1(64)

        self.decoder_rgb = decoder(512)

        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.sigmoid = nn.Sigmoid()

        self.rot_equi_att = RotEquiAttention(in_channels=1, angles=(0,15,-15,30,-30))

    def forward(self, x_rgb):
        x1_rgb = self.vgg.conv1(x_rgb)
        x2_rgb = self.vgg.conv2(x1_rgb)
        x3_rgb = self.vgg.conv3(x2_rgb)
        x4_rgb = self.vgg.conv4(x3_rgb)
        x5_rgb = self.vgg.conv5(x4_rgb)

        # up means update
        x5_MBAFM = self.MBAFM5(x4_rgb, x5_rgb)
        x4_MBAFM = self.MBAFM4(x3_rgb, x4_rgb, x5_rgb)
        x3_MBAFM = self.MBAFM3(x2_rgb, x3_rgb, x4_rgb)
        x2_MBAFM = self.MBAFM2(x1_rgb, x2_rgb, x3_rgb)
        x1_MBAFM = self.MBAFM1(x1_rgb, x2_rgb)


        s1, s2, s3, s4, s5 = self.decoder_rgb(x5_MBAFM, x4_MBAFM, x3_MBAFM, x2_MBAFM, x1_MBAFM)

        s3 = self.upsample2(s3)
        s4 = self.upsample4(s4)
        s5 = self.upsample8(s5)

        s1_att = self.rot_equi_att(s1)
        p1 = self.sigmoid(s1_att)
        p2 = self.sigmoid(s2)
        p3 = self.sigmoid(s3)
        p4 = self.sigmoid(s4)
        p5 = self.sigmoid(s5)
        return  s1, s2, s3, s4, s5, p1, p2, p3, p4, p5