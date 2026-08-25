import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torch.nn.init as init

# ==================== Default SDNet Config ====================
_default_sdnet_cfg = {
    "MODEL": {
        "MU": 0.0,
        "SQUARE_NOISE": True,
        "EXPANSION_FACTOR": 1,
        "NONEGATIVE": True,
        "NUM_LAYERS": 2,
        "WNORM": True,
        "ADAPTIVELAMBDA": False,
    },
    "DATASET": {
        "DATASET": "cifar10",
    },
}

# -------------Initialization----------------------------------------
def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                variance_scaling_initializer(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

# -------------ResNet Block (One)----------------------------------------
class Resblock(nn.Module):
    def __init__(self):
        super(Resblock, self).__init__()
        channel = 32
        self.conv20 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv21 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        rs1 = self.relu(self.conv20(x))
        rs1 = self.conv21(rs1)
        rs = torch.add(x, rs1)
        return rs

# -----------------------------------------------------
class FusionNet(nn.Module):
    def __init__(self, spectral_num=8):
        super(FusionNet, self).__init__()
        self.spectral_num = spectral_num
        hidden_channels = 11
        self.conv1 = nn.Conv2d(in_channels=spectral_num, out_channels=hidden_channels, kernel_size=3, padding=1, bias=True)
        self.depthwise = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels, bias=False)
        self.pointwise = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=hidden_channels, out_channels=spectral_num, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        init_weights(self.conv1, self.depthwise, self.pointwise, self.conv3)

    def forward(self, x, y):
        pan_concat = y.repeat(1, self.spectral_num, 1, 1)
        input_data = torch.sub(pan_concat, x)
        rs = self.relu(self.conv1(input_data))
        res = rs
        rs = self.relu(self.depthwise(rs))
        rs = self.relu(self.pointwise(rs))
        rs = rs + res
        output = self.conv3(rs)
        return output

# ==================== SDNet Core Classes (from ASDNet) ====================

class elasnet_prox(nn.Module):
    """Elastic net proximal operator."""

    def __init__(self, lambd=0.1, mu=0.0):
        super(elasnet_prox, self).__init__()
        self.lambd = lambd
        self.scaling_mu = 1.0 / (1.0 + mu)

    def forward(self, input_data):
        scaled_input = input_data * self.scaling_mu
        lambd_scaled = self.lambd * self.scaling_mu
        return self.custom_softshrink(scaled_input, lambd_scaled)

    def custom_softshrink(self, input_data, lambd):
        positive = torch.where(input_data > lambd, input_data - lambd, torch.tensor(0.0, device=input_data.device))
        negative = torch.where(input_data < -lambd, input_data + lambd, torch.tensor(0.0, device=input_data.device))
        return positive + negative


class DictBlock(nn.Module):
    """Sparse coding dictionary block (ISTA/FISTA)."""

    def __init__(self, n_channel, dict_size, mu=0.0, n_dict=1, non_negative=True,
                 stride=1, kernel_size=3, padding=1, share_weight=True, square_noise=True,
                 n_steps=10, step_size_fixed=True, step_size=0.1, w_norm=True, padding_mode="constant",
                 adaptive_lambda=False):
        super(DictBlock, self).__init__()
        self.mu = mu
        self.n_dict = n_dict
        self.stride = stride
        self.kernel_size = (kernel_size, kernel_size)
        self.padding = padding
        self.padding_mode = padding_mode
        self.groups = 1
        self.n_steps = n_steps
        self.conv_transpose_output_padding = 0 if stride == 1 else 1
        self.w_norm = w_norm
        self.non_negative = non_negative
        self.v_max = None
        self.v_max_error = 0.0
        self.xsize = None
        self.zsize = None
        self.lmbd_ = None
        self.square_noise = square_noise
        self.adaptive_lambda = adaptive_lambda

        self.weight = nn.Parameter(torch.Tensor(dict_size, self.n_dict * n_channel, kernel_size, kernel_size))
        self.lmbd = nn.Parameter(torch.tensor([0.01]), requires_grad=True)
        if hasattr(DictBlock, '_fixed_lmbd') and DictBlock._fixed_lmbd is not None:
            self.lmbd = nn.Parameter(torch.tensor([DictBlock._fixed_lmbd]), requires_grad=False)
        with torch.no_grad():
            init.kaiming_uniform_(self.weight)

        self.nonlinear = elasnet_prox(self.lmbd.item() * step_size, self.mu * step_size)
        self.register_buffer("step_size", torch.tensor(step_size, dtype=torch.float))

    def fista_forward(self, x):
        for i in range(self.n_steps):
            weight = self.weight
            step_size = self.step_size
            if i == 0:
                c_pre = 0.0
                c = step_size * F.conv2d(x.repeat(1, self.n_dict, 1, 1), weight, bias=None,
                                         stride=self.stride, padding=self.padding)
                c = self.nonlinear(c)
            elif i == 1:
                c_pre = c
                xp = F.conv_transpose2d(c, weight, bias=None, stride=self.stride, padding=self.padding,
                                        output_padding=self.conv_transpose_output_padding)
                r = x.repeat(1, self.n_dict, 1, 1) - xp
                if self.square_noise:
                    gra = F.conv2d(r, weight, bias=None, stride=self.stride, padding=self.padding)
                else:
                    w = r.view(r.size(0), -1)
                    normw = w.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12).expand_as(w).detach()
                    w = (w / normw).view(r.size())
                    gra = F.conv2d(w, weight, bias=None, stride=self.stride, padding=self.padding) * 0.5
                c = c + step_size * gra
                c = self.nonlinear(c)
                t = (math.sqrt(5.0) + 1.0) / 2.0
            else:
                t_pre = t
                t = (math.sqrt(1.0 + 4.0 * t_pre * t_pre) + 1) / 2.0
                a = (t_pre + t - 1.0) / t * c + (1.0 - t_pre) / t * c_pre
                c_pre = c
                xp = F.conv_transpose2d(c, weight, bias=None, stride=self.stride, padding=self.padding,
                                        output_padding=self.conv_transpose_output_padding)
                r = x.repeat(1, self.n_dict, 1, 1) - xp
                if self.square_noise:
                    gra = F.conv2d(r, weight, bias=None, stride=self.stride, padding=self.padding)
                else:
                    w = r.view(r.size(0), -1)
                    normw = w.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12).expand_as(w).detach()
                    w = (w / normw).view(r.size())
                    gra = F.conv2d(w, weight, bias=None, stride=self.stride, padding=self.padding) * 0.5
                c = a + step_size * gra
                c = self.nonlinear(c)
            if self.non_negative:
                c = F.relu(c)
        return c, weight

    def forward(self, x):
        # Sync nonlinear prox parameters from learnable lmbd (keeps grad path)
        self.nonlinear.lambd = self.lmbd * self.step_size
        self.nonlinear.scaling_mu = 1.0 / (1.0 + self.mu * self.step_size)
        if self.xsize is None:
            self.xsize = (x.size(-3), x.size(-2), x.size(-1))
        else:
            assert self.xsize[-3] == x.size(-3) and self.xsize[-2] == x.size(-2) and self.xsize[-1] == x.size(-1)
        if self.w_norm:
            self.normalize_weight()
        c, weight = self.fista_forward(x)
        xp = F.conv_transpose2d(c, weight, bias=None, stride=self.stride, padding=self.padding,
                                output_padding=self.conv_transpose_output_padding)
        r = x.repeat(1, self.n_dict, 1, 1) - xp
        r_loss = torch.sum(torch.pow(r, 2))/ self.n_dict
        c_loss = self.lmbd * torch.sum(torch.abs(c))+ self.mu / 2.0 * torch.sum(torch.pow(c, 2))
        if self.zsize is None:
            self.zsize = (c.size(-3), c.size(-2), c.size(-1))
        else:
            assert self.zsize[-3] == c.size(-3) and self.zsize[-2] == c.size(-2) and self.zsize[-1] == c.size(-1)
        if self.lmbd_ is None and self.adaptive_lambda:
            self.lmbd_ = self.lmbd * self.xsize[-3] * self.xsize[-2] * self.xsize[-1] / (
                self.zsize[-3] * self.zsize[-2] * self.zsize[-1]
            )
            self.lmbd = self.lmbd_
        return c, (r_loss, c_loss)

    def update_stepsize(self):
        step_size = 0.9 / self.power_iteration(self.weight)
        self.step_size = self.step_size * 0.0 + step_size
        self.nonlinear.lambd = self.lmbd * step_size
        self.nonlinear.scaling_mu = 1.0 / (1.0 + self.mu * step_size)

    def normalize_weight(self):
        with torch.no_grad():
            w = self.weight.view(self.weight.size(0), -1)
            normw = w.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12).expand_as(w)
            w = (w / normw).view(self.weight.size())
            self.weight.data = w.data

    def power_iteration(self, weight):
        max_iteration = 50
        v_max_error = 1.0e5
        tol = 1.0e-5
        k = 0
        with torch.no_grad():
            if self.v_max is None:
                c = weight.shape[0]
                v = torch.randn(size=(1, c, self.zsize[-2], self.zsize[-1])).to(weight.device)
            else:
                v = self.v_max.clone()
            while k < max_iteration and v_max_error > tol:
                tmp = F.conv_transpose2d(v, weight, bias=None, stride=self.stride, padding=self.padding,
                                         output_padding=self.conv_transpose_output_padding)
                v_ = F.conv2d(tmp, weight, bias=None, stride=self.stride, padding=self.padding)
                v_ = F.normalize(v_.view(-1), dim=0, p=2).view(v.size())
                v_max_error = torch.sum((v_ - v) ** 2)
                k += 1
                v = v_
            v_max = v.clone()
            Dv_max = F.conv_transpose2d(v_max, weight, bias=None, stride=self.stride, padding=self.padding,
                                        output_padding=self.conv_transpose_output_padding)
            lambda_max = torch.sum(Dv_max ** 2).item()
        self.v_max = v_max
        return lambda_max


class DictConv2d(nn.Module):
    """2D convolution wrapper around DictBlock for SDNet."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, cfg=None):
        super(DictConv2d, self).__init__()
        if cfg is None:
            cfg = _default_sdnet_cfg
        self.dn = DictBlock(
            in_channels, out_channels, stride=stride, kernel_size=kernel_size, padding=padding,
            mu=cfg["MODEL"]["MU"], square_noise=cfg["MODEL"]["SQUARE_NOISE"],
            n_dict=cfg["MODEL"]["EXPANSION_FACTOR"], non_negative=cfg["MODEL"]["NONEGATIVE"],
            n_steps=cfg["MODEL"]["NUM_LAYERS"], w_norm=cfg["MODEL"]["WNORM"],
            adaptive_lambda=cfg["MODEL"].get("ADAPTIVELAMBDA", False),
        )
        self.rc = None
        self.r_loss = []

    def get_rc(self):
        if self.rc is None:
            raise ValueError("should call forward first.")
        return self.rc

    def forward(self, x):
        device = x.device
        out, rc = self.dn(x.to(device))
        self.rc = rc
        if self.training is False:
            self.r_loss.extend([self.rc[0].item() / len(x)] * len(x))
        return out


class DictConv2d_all(nn.Module):
    """DictConv2d for all-CSC architecture. No rc tracking, simpler."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, cfg=None):
        super(DictConv2d_all, self).__init__()
        if cfg is None:
            cfg = _default_sdnet_cfg
        self.in_stride = stride
        self.stride = 1 if cfg["MODEL"].get("POOLING", False) else stride
        self.dn = DictBlock(
            in_channels, out_channels, stride=self.stride, kernel_size=kernel_size, padding=padding,
            mu=cfg["MODEL"]["MU"], square_noise=cfg["MODEL"]["SQUARE_NOISE"],
            n_dict=cfg["MODEL"]["EXPANSION_FACTOR"], non_negative=cfg["MODEL"]["NONEGATIVE"],
            n_steps=cfg["MODEL"]["NUM_LAYERS"], w_norm=cfg["MODEL"]["WNORM"],
            adaptive_lambda=cfg["MODEL"].get("ADAPTIVELAMBDA", False),
        )

    def forward(self, x):
        out, rc = self.dn(x)
        return out


class SDNetFusionNet_All(nn.Module):
    """Pansharpening fusion model where ALL conv layers are DictConv2d (CSC layers).
    No ReLU - DictBlock handles nonlinearity internally."""

    def __init__(self, spectral_num=8, cfg=None):
        super(SDNetFusionNet_All, self).__init__()
        if cfg is None:
            cfg = _default_sdnet_cfg
        self.cfg = cfg
        self.spectral_num = spectral_num
        hidden = 11

        self.conv_in = DictConv2d_all(spectral_num, hidden, kernel_size=3, padding=1, bias=False, cfg=cfg)
        self.bn_in = nn.BatchNorm2d(hidden)

        self.csc1 = DictConv2d_all(hidden, hidden, kernel_size=3, padding=1, bias=False, cfg=cfg)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.csc2 = DictConv2d_all(hidden, hidden, kernel_size=3, padding=1, bias=False, cfg=cfg)
        self.bn2 = nn.BatchNorm2d(hidden)

        self.csc3 = DictConv2d_all(hidden, hidden, kernel_size=3, padding=1, bias=False, cfg=cfg)
        self.bn3 = nn.BatchNorm2d(hidden)
        self.csc4 = DictConv2d_all(hidden, hidden, kernel_size=3, padding=1, bias=False, cfg=cfg)
        self.bn4 = nn.BatchNorm2d(hidden)

        self.conv_out = nn.Conv2d(hidden, spectral_num, kernel_size=3, padding=1, bias=True)
        init_weights(self.conv_out)

    def update_stepsize(self):
        for m in self.modules():
            if isinstance(m, DictBlock):
                m.update_stepsize()

    def forward(self, x, y):
        pan_concat = y.repeat(1, self.spectral_num, 1, 1)
        input_data = torch.sub(pan_concat, x)

        out = self.bn_in(self.conv_in(input_data))
        residual1 = out
        out = self.bn2(self.csc2(self.bn1(self.csc1(out))))
        out = out + residual1

        residual2 = out
        out = self.bn4(self.csc4(self.bn3(self.csc3(out))))
        out = out + residual2

        return self.conv_out(out)


# ==================== Ablation: Pure Conv (no CSC) ====================
class ConvBlock(nn.Module):
    """Plain Conv2d + BN + ReLU, same interface as DictConv2d_all for ablation."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SDNetFusionNet_Conv(nn.Module):
    """Same architecture as SDNetFusionNet_All, but all CSC layers replaced with ConvBlock.
    Used for ablation: measuring contribution of sparse coding (DictBlock)."""

    def __init__(self, spectral_num=8):
        super(SDNetFusionNet_Conv, self).__init__()
        self.spectral_num = spectral_num
        hidden = 11

        self.conv_in = ConvBlock(spectral_num, hidden, kernel_size=3, padding=1, bias=False)
        self.csc1 = ConvBlock(hidden, hidden, kernel_size=3, padding=1, bias=False)
        self.csc2 = ConvBlock(hidden, hidden, kernel_size=3, padding=1, bias=False)
        self.csc3 = ConvBlock(hidden, hidden, kernel_size=3, padding=1, bias=False)
        self.csc4 = ConvBlock(hidden, hidden, kernel_size=3, padding=1, bias=False)
        self.conv_out = nn.Conv2d(hidden, spectral_num, kernel_size=3, padding=1, bias=True)
        init_weights(self.conv_in.conv, self.csc1.conv, self.csc2.conv, self.csc3.conv, self.csc4.conv, self.conv_out)

    def forward(self, x, y):
        pan_concat = y.repeat(1, self.spectral_num, 1, 1)
        input_data = torch.sub(pan_concat, x)

        out = self.conv_in(input_data)
        residual1 = out
        out = self.csc2(self.csc1(out))
        out = out + residual1

        residual2 = out
        out = self.csc4(self.csc3(out))
        out = out + residual2

        return self.conv_out(out)


class SDNetBasicBlock(nn.Module):
    """Basic residual block used in SDNet18."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(SDNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SDNetResNet(nn.Module):
    """SDNet ResNet backbone used by SDNet18."""

    def __init__(self, block, num_blocks, num_classes=10, Dataname=None, cfg=None):
        super(SDNetResNet, self).__init__()
        if cfg is None:
            cfg = _default_sdnet_cfg
        self.in_planes = 64
        self.cfg = cfg

        self.layer_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer_2 = nn.BatchNorm2d(64)

        if "cifar" in Dataname:
            self.layer0 = nn.Sequential(
                DictConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False, cfg=cfg),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
        elif "imagenet" in Dataname:
            self.layer0 = nn.Sequential(
                DictConv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False, cfg=cfg),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        else:
            raise ValueError(f"Unknown Dataname: {Dataname}")

        self.layer_4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer_5 = nn.BatchNorm2d(64)
        self.layer_6 = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def update_stepsize(self):
        for m in self.modules():
            if isinstance(m, DictBlock):
                m.update_stepsize()

    def get_rc(self):
        rc_list = []
        for m in self.modules():
            if isinstance(m, DictConv2d):
                rc_list.append(m.get_rc())
        return rc_list

    def forward(self, x):
        device = x.device
        out = self.layer_1(x.to(device))
        out = self.layer_2(out)
        out = self.layer0(out)
        second_layer_rc = self.layer0[0].get_rc()
        out = self.layer_4(out)
        out = self.layer_5(out)
        out = self.layer_6(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, second_layer_rc


def SDNet18(num_classes, cfg=None):
    """Factory function for SDNet18 model."""
    if cfg is None:
        cfg = _default_sdnet_cfg
    return SDNetResNet(SDNetBasicBlock, [2, 2, 2, 1], num_classes, cfg["DATASET"]["DATASET"], cfg=cfg)


# ==================== SDNet-based Fusion Network ====================

class SDNetFusionNet(nn.Module):
    """Pansharpening fusion model using SDNet-style DictConv2d sparse coding layers."""

    def __init__(self, spectral_num=8, cfg=None):
        super(SDNetFusionNet, self).__init__()
        if cfg is None:
            cfg = _default_sdnet_cfg
        self.cfg = cfg
        self.spectral_num = spectral_num
        hidden_channels = 11

        # Input projection
        self.conv_in = nn.Conv2d(spectral_num, hidden_channels, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(hidden_channels)

        # Core sparse coding layer (DictConv2d replaces standard convs)
        self.dict_conv = DictConv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=False, cfg=cfg)
        self.bn_dict = nn.BatchNorm2d(hidden_channels)

        # Additional feature refinement
        self.conv_mid = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False)
        self.bn_mid = nn.BatchNorm2d(hidden_channels)

        # Output projection
        self.conv_out = nn.Conv2d(hidden_channels, spectral_num, kernel_size=3, padding=1, bias=True)

        self.relu = nn.ReLU(inplace=True)

        # Initialize weights
        init_weights(self.conv_in, self.conv_mid, self.conv_out)

    def update_stepsize(self):
        for m in self.modules():
            if isinstance(m, DictBlock):
                m.update_stepsize()

    def get_rc(self):
        rc_list = []
        for m in self.modules():
            if isinstance(m, DictConv2d):
                rc_list.append(m.get_rc())
        return rc_list

    def forward(self, x, y):
        # Preprocessing: same as FusionNet
        pan_concat = y.repeat(1, self.spectral_num, 1, 1)
        input_data = torch.sub(pan_concat, x)

        # Feature extraction with DictConv2d
        out = self.relu(self.bn_in(self.conv_in(input_data)))
        residual = out
        out = self.relu(self.bn_dict(self.dict_conv(out)))
        out = self.relu(self.bn_mid(self.conv_mid(out)))
        out = out + residual
        output = self.conv_out(out)

        return output


# ----------------- End-Main-Part ------------------------------------
def variance_scaling_initializer(tensor):
    from scipy.stats import truncnorm

    def truncated_normal_(tensor, mean=0, std=1):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor

    def variance_scaling(x, scale=1.0, mode="fan_in", distribution="truncated_normal", seed=None):
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(x)
        if mode == "fan_in":
            scale /= max(1.0, fan_in)
        elif mode == "fan_out":
            scale /= max(1.0, fan_out)
        else:
            scale /= max(1.0, (fan_in + fan_out) / 2.0)
        if distribution == "normal" or distribution == "truncated_normal":
            stddev = math.sqrt(scale) / 0.87962566103423978
        truncated_normal_(x, 0.0, stddev)
        return x / 10 * 1.28

    variance_scaling(tensor)
    return tensor


def summaries(model, writer=None, grad=False):
    if grad:
        from torchsummary import summary
        summary(model, input_size=[(8, 64, 64), (1, 64, 64)], batch_size=1)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)
    if writer is not None:
        x = torch.randn(1, 64, 64, 64)
        writer.add_graph(model, (x,))
