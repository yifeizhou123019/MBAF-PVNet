import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ======================== DW+PW（深度可分离卷积）封装 ========================
class LiteConv2d(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super().__init__()
        assert kernel_size == 3, "LiteConv2d 仅用于替换 3x3 卷积"
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding,
                            dilation=dilation, groups=in_ch, bias=bias)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        if kernel_size == 3:
            self.op = LiteConv2d(in_planes, out_planes, 3, stride, padding, dilation, bias=False)
        else:
            self.op = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride,
                          padding=padding, dilation=dilation, bias=False),
                nn.BatchNorm2d(out_planes),
                nn.ReLU(inplace=True)
            )
    def forward(self, x):
        return self.op(x)

class Up2(nn.Module):
    """双线性上采样 ×2 + DW+PW(3x3) 清理锯齿，同时轻量化。"""
    def __init__(self, channels):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.dw(x); x = self.pw(x); x = self.bn(x)
        return self.act(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = self.conv1(max_out)
        return self.sigmoid(x)

class DLCM(nn.Module):
    def __init__(self, in_channels, phi='T'):
        super().__init__()
        Kernel_size = {'T': 5, 'B': 7, 'S': 5, 'L': 7}[phi]
        groups = {'T': in_channels, 'B': in_channels, 'S': in_channels // 8, 'L': in_channels // 8}[phi]
        def get_valid_group(in_channels):
            for g in [32,16,8,4,2,1]:
                if in_channels % g == 0: return g
            return 1
        num_groups = get_valid_group(in_channels)
        pad = Kernel_size // 2
        self.conv1_h = nn.Conv1d(in_channels, in_channels, Kernel_size, padding=pad, groups=groups, bias=False)
        self.conv1_w = nn.Conv1d(in_channels, in_channels, Kernel_size, padding=pad, groups=groups, bias=False)
        self.gn = nn.GroupNorm(num_groups, in_channels)
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self,x):
        b,c,h,w = x.size()
        x_h = torch.mean(x, dim=3, keepdim=True).view(b,c,h)
        x_w = torch.mean(x, dim=2, keepdim=True).view(b,c,w)
        x_h = F.silu(self.gn(self.conv1_h(x_h))).view(b,c,h,1)
        x_w = F.silu(self.gn(self.conv1_w(x_w))).view(b,c,1,w)
        spatial_att = x_h * x_w
        channel_att = self.channel_att(x)
        return x * (1 + spatial_att * channel_att)

class CCAM(nn.Module):
    def __init__(self, dim=512, kernel_size=3):
        super().__init__()
        self.dim = dim; self.kernel_size = kernel_size
        # 保持你原来的 groups=4 设计（不是 depthwise），不做 DW+PW 改动
        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=4, bias=False),
            nn.BatchNorm2d(dim), nn.ReLU()
        )
        self.value_embed = nn.Sequential(nn.Conv2d(dim, dim, 1, bias=False), nn.BatchNorm2d(dim))
        factor = 4
        self.attention_embed = nn.Sequential(
            nn.Conv2d(2*dim, 2*dim//factor, 1, bias=False), nn.BatchNorm2d(2*dim//factor), nn.ReLU(),
            nn.Conv2d(2*dim//factor, kernel_size*kernel_size*dim, 1)
        )
    def forward(self,x):
        bs,c,h,w = x.shape
        k1 = self.key_embed(x)
        v  = self.value_embed(x).view(bs,c,-1)
        y  = torch.cat([k1,x], dim=1)
        att= self.attention_embed(y).reshape(bs,c,self.kernel_size*self.kernel_size,h,w)
        att= att.mean(2).view(bs,c,-1)
        k2 = (F.softmax(att, dim=-1) * v).view(bs,c,h,w)
        return k1 + k2

class CCAM_DLCM_Gated(nn.Module):
    def __init__(self, in_channels, phi='T', kernel_size=3):
        super().__init__()
        self.dlcm = DLCM(in_channels, phi)
        self.cca  = CCAM(in_channels, kernel_size)
        self.gate = nn.Sequential(nn.Conv2d(in_channels*2, in_channels, 1, bias=False),
                                  nn.BatchNorm2d(in_channels), nn.Sigmoid())
    def forward(self,x):
        dlcm_out = self.dlcm(x); cca_out = self.cca(x)
        fusion = torch.cat([dlcm_out, cca_out], dim=1)
        g = self.gate(fusion)
        return g * cca_out + (1 - g) * dlcm_out

class RotEquiAttention(nn.Module):
    def __init__(self, in_channels, mid_channels=None, angles=(0, 45, 90, 135)):
        super().__init__()
        self.in_c = in_channels
        self.mid_c = mid_channels if mid_channels is not None else max(in_channels // 4, 1)
        self.angles = angles
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.in_c, self.mid_c, 1, bias=False),
            nn.BatchNorm2d(self.mid_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_c, self.in_c, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(nn.Conv2d(2,1,7,padding=3,bias=False), nn.Sigmoid())
    def forward(self, F_in):
        B,C,H,W = F_in.size()
        device = F_in.device
        fused = torch.zeros_like(F_in)
        for ang in self.angles:
            theta = torch.tensor([
                [torch.cos(torch.deg2rad(torch.tensor(ang))), -torch.sin(torch.deg2rad(torch.tensor(ang))), 0],
                [torch.sin(torch.deg2rad(torch.tensor(ang))),  torch.cos(torch.deg2rad(torch.tensor(ang))), 0]
            ], device=device).unsqueeze(0).repeat(B,1,1)
            grid  = F.affine_grid(theta, F_in.size(), align_corners=True)
            F_rot = F.grid_sample(F_in, grid, mode='bilinear', align_corners=True)

            ca = self.channel_att(F_rot)
            F_ca = F_rot * ca
            avg_out = torch.mean(F_ca, dim=1, keepdim=True)
            max_out,_ = torch.max(F_ca, dim=1, keepdim=True)
            sa = self.spatial_att(torch.cat([avg_out, max_out], dim=1))
            F_att = F_ca * sa

            inv_ang = -ang
            inv_theta = torch.tensor([
                [torch.cos(torch.deg2rad(torch.tensor(inv_ang))), -torch.sin(torch.deg2rad(torch.tensor(inv_ang))), 0],
                [torch.sin(torch.deg2rad(torch.tensor(inv_ang))),  torch.cos(torch.deg2rad(torch.tensor(inv_ang))), 0]
            ], device=device).unsqueeze(0).repeat(B,1,1)
            inv_grid = F.affine_grid(inv_theta, F_att.size(), align_corners=True)
            F_back = F.grid_sample(F_att, inv_grid, mode='bilinear', align_corners=True)
            fused += F_back
        return fused / float(len(self.angles))

# ======================== MBAFM 模块（分支保持原样，仅 3x3 变轻） ==============
class MBAFM5(nn.Module):
    def __init__(self, cur_channel):
        super().__init__()
        self.cur_b1 = BasicConv2d(cur_channel, cur_channel, 3, padding=1, dilation=1)
        self.cur_b2 = BasicConv2d(cur_channel, cur_channel, 3, padding=2, dilation=2)
        self.cur_b3 = BasicConv2d(cur_channel, cur_channel, 3, padding=3, dilation=3)
        self.cur_b4 = BasicConv2d(cur_channel, cur_channel, 3, padding=4, dilation=4)
        self.cur_all = BasicConv2d(4*cur_channel, cur_channel, 3, padding=1)
        self.cca_dlcm = CCAM_DLCM_Gated(cur_channel, phi='T', kernel_size=3)
        self.pre_sa = SpatialAttention()
    def forward(self, x_pre, x_cur):
        x_all = self.cur_all(torch.cat([self.cur_b1(x_cur), self.cur_b2(x_cur),
                                        self.cur_b3(x_cur), self.cur_b4(x_cur)], dim=1))
        fused = self.cca_dlcm(x_all)
        x_pre_ds = F.interpolate(x_pre, size=x_all.shape[2:], mode='bilinear', align_corners=True)
        pre_part = x_all * self.pre_sa(x_pre_ds)
        return fused + pre_part + x_cur

class MBAFM4(nn.Module):
    def __init__(self, cur_channel):
        super().__init__()
        self.cur_b1 = BasicConv2d(cur_channel, cur_channel, 3, padding=1, dilation=1)
        self.cur_b2 = BasicConv2d(cur_channel, cur_channel, 3, padding=2, dilation=2)
        self.cur_b3 = BasicConv2d(cur_channel, cur_channel, 3, padding=3, dilation=3)
        self.cur_b4 = BasicConv2d(cur_channel, cur_channel, 3, padding=4, dilation=4)
        self.cur_all = BasicConv2d(4*cur_channel, cur_channel, 3, padding=1)
        self.cca_dlcm = CCAM_DLCM_Gated(cur_channel, phi='T', kernel_size=3)
        self.pre_sa = SpatialAttention(); self.lat_sa = SpatialAttention()
    def forward(self, x_pre, x_cur, x_lat):
        x_all = self.cur_all(torch.cat([self.cur_b1(x_cur), self.cur_b2(x_cur),
                                        self.cur_b3(x_cur), self.cur_b4(x_cur)], dim=1))
        fused = self.cca_dlcm(x_all)
        x_pre_ds = F.interpolate(x_pre, size=x_all.shape[2:], mode='bilinear', align_corners=True)
        x_lat_us = F.interpolate(x_lat, size=x_all.shape[2:], mode='bilinear', align_corners=True)
        pre_part = x_all * self.pre_sa(x_pre_ds)
        lat_part = x_all * self.lat_sa(x_lat_us)
        return fused + pre_part + lat_part + x_cur

class MBAFM3(nn.Module):
    def __init__(self, cur_channel):
        super().__init__()
        self.cur_b1 = BasicConv2d(cur_channel, cur_channel, 3, padding=1, dilation=1)
        self.cur_b2 = BasicConv2d(cur_channel, cur_channel, 3, padding=2, dilation=2)
        self.cur_b3 = BasicConv2d(cur_channel, cur_channel, 3, padding=3, dilation=3)
        self.cur_b4 = BasicConv2d(cur_channel, cur_channel, 3, padding=4, dilation=4)
        self.cur_all = BasicConv2d(4*cur_channel, cur_channel, 3, padding=1)
        self.cca_dlcm = CCAM_DLCM_Gated(cur_channel, phi='T', kernel_size=3)
        self.pre_sa = SpatialAttention(); self.lat_sa = SpatialAttention()
    def forward(self, x_pre, x_cur, x_lat):
        x_all = self.cur_all(torch.cat([self.cur_b1(x_cur), self.cur_b2(x_cur),
                                        self.cur_b3(x_cur), self.cur_b4(x_cur)], dim=1))
        fused = self.cca_dlcm(x_all)
        x_pre_ds = F.interpolate(x_pre, size=x_all.shape[2:], mode='bilinear', align_corners=True)
        x_lat_us = F.interpolate(x_lat, size=x_all.shape[2:], mode='bilinear', align_corners=True)
        pre_part = x_all * self.pre_sa(x_pre_ds)
        lat_part = x_all * self.lat_sa(x_lat_us)
        return fused + pre_part + lat_part + x_cur

class MBAFM2(nn.Module):
    def __init__(self, cur_channel):
        super().__init__()
        self.cur_b1 = BasicConv2d(cur_channel, cur_channel, 3, padding=1, dilation=1)
        self.cur_b2 = BasicConv2d(cur_channel, cur_channel, 3, padding=2, dilation=2)
        self.cur_b3 = BasicConv2d(cur_channel, cur_channel, 3, padding=3, dilation=3)
        self.cur_b4 = BasicConv2d(cur_channel, cur_channel, 3, padding=4, dilation=4)
        self.cur_all = BasicConv2d(4*cur_channel, cur_channel, 3, padding=1)
        self.dlcm   = DLCM(cur_channel, phi='T')
        self.pre_sa = SpatialAttention(); self.lat_sa = SpatialAttention()
    def forward(self, x_pre, x_cur, x_lat):
        x_all = self.cur_all(torch.cat([self.cur_b1(x_cur), self.cur_b2(x_cur),
                                        self.cur_b3(x_cur), self.cur_b4(x_cur)], dim=1))
        fused = self.dlcm(x_all)
        x_pre_ds = F.interpolate(x_pre, size=x_all.shape[2:], mode='bilinear', align_corners=True)
        x_lat_us = F.interpolate(x_lat, size=x_all.shape[2:], mode='bilinear', align_corners=True)
        pre_part = x_all * self.pre_sa(x_pre_ds)
        lat_part = x_all * self.lat_sa(x_lat_us)
        return fused + pre_part + lat_part + x_cur

class MBAFM1(nn.Module):
    def __init__(self, cur_channel):
        super().__init__()
        self.cur_b1 = BasicConv2d(cur_channel, cur_channel, 3, padding=1, dilation=1)
        self.cur_b2 = BasicConv2d(cur_channel, cur_channel, 3, padding=2, dilation=2)
        self.cur_b3 = BasicConv2d(cur_channel, cur_channel, 3, padding=3, dilation=3)
        self.cur_b4 = BasicConv2d(cur_channel, cur_channel, 3, padding=4, dilation=4)
        self.cur_all = BasicConv2d(4*cur_channel, cur_channel, 3, padding=1)
        self.dlcm   = DLCM(cur_channel, phi='T')
        self.lat_sa = SpatialAttention()
    def forward(self, x_cur, x_lat):
        x_all = self.cur_all(torch.cat([self.cur_b1(x_cur), self.cur_b2(x_cur),
                                        self.cur_b3(x_cur), self.cur_b4(x_cur)], dim=1))
        fused = self.dlcm(x_all)
        x_lat_us = F.interpolate(x_lat, size=x_all.shape[2:], mode='bilinear', align_corners=True)
        lat_part = x_all * self.lat_sa(x_lat_us)
        return fused + lat_part + x_cur

# ======================== BAB + 解码器（所有 3x3 已轻量化） ====================
class BAB_Decoder(nn.Module):
    def __init__(self, channel_1=1024, channel_2=512, channel_3=256, dilation_1=3, dilation_2=2):
        super().__init__()
        self.conv1      = BasicConv2d(channel_1, channel_2, 3, padding=1)
        self.conv1_Dila = BasicConv2d(channel_2, channel_2, 3, padding=dilation_1, dilation=dilation_1)
        self.conv2      = BasicConv2d(channel_2, channel_2, 3, padding=1)
        self.conv2_Dila = BasicConv2d(channel_2, channel_2, 3, padding=dilation_2, dilation=dilation_2)
        self.conv3      = BasicConv2d(channel_2, channel_2, 3, padding=1)
        self.conv_fuse  = BasicConv2d(channel_2*3, channel_3, 3, padding=1)
    def forward(self, x):
        x1 = self.conv1(x); x1_d = self.conv1_Dila(x1)
        x2 = self.conv2(x1); x2_d = self.conv2_Dila(x2)
        x3 = self.conv3(x2)
        return self.conv_fuse(torch.cat([x1_d, x2_d, x3], dim=1))

class decoder(nn.Module):
    def __init__(self, channel=512):
        super().__init__()
        self.decoder5 = nn.Sequential(BAB_Decoder(512, 512, 512, 3, 2), nn.Dropout(0.5), Up2(512))
        self.S5 = nn.Conv2d(512, 1, 3, padding=1)

        self.decoder4 = nn.Sequential(BAB_Decoder(1024, 512, 256, 3, 2), nn.Dropout(0.5), Up2(256))
        self.S4 = nn.Conv2d(256, 1, 3, padding=1)

        self.decoder3 = nn.Sequential(BAB_Decoder(512, 256, 128, 5, 3), nn.Dropout(0.5), Up2(128))
        self.S3 = nn.Conv2d(128, 1, 3, padding=1)

        self.decoder2 = nn.Sequential(BAB_Decoder(256, 128, 64, 5, 3), nn.Dropout(0.5), Up2(64))
        self.S2 = nn.Conv2d(64, 1, 3, padding=1)

        self.decoder1 = BAB_Decoder(128, 64, 32, 5, 3)
        self.up_to_full = Up2(32)  # 仍保持上次的全分辨率输出
        self.S1 = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, x5, x4, x3, x2, x1):
        # 32->16
        x5_up = self.decoder5(x5)
        if x5_up.shape[2:] != x4.shape[2:]:
            x5_up = F.interpolate(x5_up, size=x4.shape[2:], mode='bilinear', align_corners=True)
        s5 = self.S5(x5_up)

        # 16->8
        x4_in = torch.cat([x4, x5_up], dim=1)
        x4_up = self.decoder4(x4_in)
        if x4_up.shape[2:] != x3.shape[2:]:
            x4_up = F.interpolate(x4_up, size=x3.shape[2:], mode='bilinear', align_corners=True)
        s4 = self.S4(x4_up)

        # 8->4
        x3_in = torch.cat([x3, x4_up], dim=1)
        x3_up = self.decoder3(x3_in)
        if x3_up.shape[2:] != x2.shape[2:]:
            x3_up = F.interpolate(x3_up, size=x2.shape[2:], mode='bilinear', align_corners=True)
        s3 = self.S3(x3_up)

        # 4->2
        x2_in = torch.cat([x2, x3_up], dim=1)
        x2_up = self.decoder2(x2_in)
        if x2_up.shape[2:] != x1.shape[2:]:
            x2_up = F.interpolate(x2_up, size=x1.shape[2:], mode='bilinear', align_corners=True)
        s2 = self.S2(x2_up)

        # 2->1（保持 1× 原图分辨率输出）
        x1_in = torch.cat([x1, x2_up], dim=1)
        x1_up = self.decoder1(x1_in)
        x1_up = self.up_to_full(x1_up)
        s1 = self.S1(x1_up)

        return s1, s2, s3, s4, s5


class MBAF_PVNet_MobileNetV2(nn.Module):
    def __init__(self, channel=32, mobilenet_weights_path=None, use_torchvision_pretrained=True):
        super().__init__()
        try:
            from torchvision.models import MobileNet_V2_Weights
            weights = MobileNet_V2_Weights.IMAGENET1K_V1 if (use_torchvision_pretrained and mobilenet_weights_path is None) else None
            self.mobilenet_v2 = models.mobilenet_v2(weights=weights)
        except Exception:
            self.mobilenet_v2 = models.mobilenet_v2(pretrained=(use_torchvision_pretrained and mobilenet_weights_path is None))
        if mobilenet_weights_path:
            state = torch.load(mobilenet_weights_path, map_location='cpu')
            self.mobilenet_v2.load_state_dict(state, strict=False)

        self.backbone = self.mobilenet_v2.features

        # MobileNetV2 切片通道：x1≈24, x2≈32, x3≈64, x4≈96, x5=1280
        self.cvt1 = nn.Conv2d(24,    64,  1)
        self.cvt2 = nn.Conv2d(32,   128,  1)
        self.cvt3 = nn.Conv2d(64,   256,  1)
        self.cvt4 = nn.Conv2d(96,   512,  1)
        self.cvt5 = nn.Conv2d(1280, 512,  1)

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
        self.rot_equi_att = RotEquiAttention(in_channels=1, angles=(0, 15, -15, 30, -30))

    def forward(self, x_rgb):
        x1 = self.backbone[0:4](x_rgb)    # 1/2,  C≈24
        x2 = self.backbone[4:7](x1)       # 1/4,  C≈32
        x3 = self.backbone[7:11](x2)      # 1/8,  C≈64
        x4 = self.backbone[11:14](x3)     # 1/16, C≈96
        x5 = self.backbone[14:](x4)       # 1/32, C=1280

        x1 = self.cvt1(x1); x2 = self.cvt2(x2); x3 = self.cvt3(x3); x4 = self.cvt4(x4); x5 = self.cvt5(x5)

        f5 = self.MBAFM5(x4, x5)
        f4 = self.MBAFM4(x3, x4, x5)
        f3 = self.MBAFM3(x2, x3, x4)
        f2 = self.MBAFM2(x1, x2, x3)
        f1 = self.MBAFM1(x1, x2)

        s1, s2, s3, s4, s5 = self.decoder_rgb(f5, f4, f3, f2, f1)
        s3 = self.upsample2(s3); s4 = self.upsample4(s4); s5 = self.upsample8(s5)

        s1_att = self.rot_equi_att(s1)
        p1 = self.sigmoid(s1_att); p2 = self.sigmoid(s2); p3 = self.sigmoid(s3); p4 = self.sigmoid(s4); p5 = self.sigmoid(s5)
        return s1, s2, s3, s4, s5, p1, p2, p3, p4, p5
