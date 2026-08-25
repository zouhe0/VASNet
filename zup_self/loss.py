import torch
import torch.nn.functional as F
import numpy as np
from wald_utilities import genMTF, MTF_PAN, fspecial_gauss
import math
import scipy.ndimage as ndimage
from scipy import signal
import kornia.filters as kf
import torch.nn as nn


class LossCalculator:
    def __init__(self, sensor, ratio, N=41, device='cpu'):
        self.sensor = sensor.upper()
        self.ratio = ratio
        self.N = N
        self.device = device

        mtf_kernel_np = genMTF(self.ratio, self.sensor, self.N)
        self.mtf_kernel = torch.from_numpy(mtf_kernel_np).float().to(device)

        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                    device=device, dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                    device=device, dtype=torch.float32).view(1, 1, 3, 3)

        self.gaussian_kernels = []
        for sigma in [0.5, 1.0, 2.0]:
            self.gaussian_kernels.append(self._create_gaussian_kernel(5, sigma).to(device))

        self.laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                                           device=device, dtype=torch.float32).view(1, 1, 3, 3)

    def _create_gaussian_kernel(self, kernel_size=5, sigma=1.0):
        coords = torch.arange(kernel_size, dtype=torch.float32)
        coords -= (kernel_size - 1) / 2

        x = coords.repeat(kernel_size, 1)
        y = x.t()

        kernel = torch.exp(-(x.pow(2) + y.pow(2)) / (2 * sigma ** 2))
        kernel /= kernel.sum()

        return kernel.view(1, 1, kernel_size, kernel_size)

    @staticmethod
    def apply_convolution(image, kernel):
        image = image.permute(2, 0, 1).unsqueeze(0)
        bands = image.shape[1]
        kH, kW = kernel.shape[0], kernel.shape[1]
        if kernel.shape[2] < bands:
            extra = kernel[:, :, -1:].repeat(1, 1, bands - kernel.shape[2])
            kernel = torch.cat([kernel, extra], dim=2)
        kernel = kernel.permute(2, 0, 1).unsqueeze(1)
        filtered = F.conv2d(image, kernel, padding=kH // 2, groups=bands)
        filtered = filtered.squeeze(0).permute(1, 2, 0)
        return filtered

    def compute_spectral_loss(self, X, Y):
        X = X.to(self.device)
        Y = Y.to(self.device)

        X_t = X.permute(2, 0, 1).unsqueeze(0)

        mtf_kernel = self.mtf_kernel
        MTF_kern = mtf_kernel.permute(2, 0, 1).unsqueeze(1)

        bands = X_t.shape[1]
        depthconv = nn.Conv2d(in_channels=bands,
                            out_channels=bands,
                            kernel_size=MTF_kern.shape[2:],
                            groups=bands,
                            padding=mtf_kernel.shape[0]//2,
                            padding_mode='replicate',
                            bias=False).to(self.device)

        depthconv.weight.data = MTF_kern
        depthconv.weight.requires_grad = False

        X_blurred = depthconv(X_t)
        X_down_bicubic = F.interpolate(X_blurred, scale_factor=1/self.ratio, mode='bicubic', align_corners=False)
        X_down = X_down_bicubic.squeeze(0).permute(1, 2, 0)

        loss = torch.norm(X_down - Y, p='fro') ** 2

        return loss

    def extract_multiscale_gradients(self, image):
        if image.dim() > 4:
            image = image.view(1, 1, image.shape[-2], image.shape[-1])
        elif image.dim() == 3:
            if image.shape[2] > 1:
                weights = torch.tensor([0.299, 0.587, 0.114], device=self.device)
                if image.shape[2] > 3:
                    extra_weights = torch.ones(image.shape[2] - 3, device=self.device) / (image.shape[2] - 3)
                    weights = torch.cat([weights, extra_weights])
                image = torch.sum(image * weights.view(1, 1, -1), dim=2)
            else:
                image = image.squeeze(2)
            image = image.unsqueeze(0).unsqueeze(0)
        elif image.dim() == 2:
            image = image.unsqueeze(0).unsqueeze(0)

        assert image.dim() == 4 and image.shape[1] == 1, f"Unexpected image shape, expected [1,1,H,W], got {image.shape}"

        gradients_list = []

        for gaussian_kernel in self.gaussian_kernels:
            smoothed = F.conv2d(image, gaussian_kernel, padding=2)

            grad_x = F.conv2d(smoothed, self.sobel_x, padding=1)
            grad_y = F.conv2d(smoothed, self.sobel_y, padding=1)

            grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
            grad_direction = torch.atan2(grad_y, grad_x)

            gradient_features = torch.cat([grad_magnitude, grad_direction], dim=1)
            gradients_list.append(gradient_features)

            laplacian = F.conv2d(smoothed, self.laplacian_kernel, padding=1)
            gradients_list.append(laplacian)

        return gradients_list

    def structure_tensor_features(self, image):
        grad_x = F.conv2d(image, self.sobel_x, padding=1)
        grad_y = F.conv2d(image, self.sobel_y, padding=1)

        Ixx = grad_x * grad_x
        Iyy = grad_y * grad_y
        Ixy = grad_x * grad_y

        kernel = self.gaussian_kernels[1]
        Jxx = F.conv2d(Ixx, kernel, padding=2)
        Jyy = F.conv2d(Iyy, kernel, padding=2)
        Jxy = F.conv2d(Ixy, kernel, padding=2)

        trace = Jxx + Jyy
        delta = torch.sqrt((Jxx - Jyy)**2 + 4 * Jxy**2 + 1e-10)

        lambda1 = (trace + delta) / 2
        lambda2 = (trace - delta) / 2

        anisotropy = (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-10)
        coherence = (lambda1 - lambda2)**2 / (lambda1 + lambda2 + 1e-10)**2

        return torch.cat([lambda1, lambda2, anisotropy, coherence], dim=1)

    def compute_guided_spatial_loss(self, X, I_PAN, lambda_gradient=1.0, lambda_structure=0.5, lambda_texture=0.5):
        if X.dim() == 4:
            X = X.squeeze(0)

        if I_PAN.dim() == 3 and I_PAN.shape[0] == 1:
            I_PAN = I_PAN.squeeze(0)
        elif I_PAN.dim() == 4:
            I_PAN = I_PAN.squeeze(0).squeeze(0)

        if self.sensor == 'WV3' or self.sensor == 'WV2':
            if X.shape[2] >= 8:
                weights = torch.tensor([0.05, 0.05, 0.1, 0.2, 0.35, 0.15, 0.05, 0.05], device=self.device)
                if X.shape[2] > 8:
                    extra = torch.ones(X.shape[2] - 8, device=self.device) * 0.05
                    weights = torch.cat([weights, extra])
            else:
                weights = torch.ones(X.shape[2], device=self.device) / X.shape[2]
        elif self.sensor == 'QB':
            weights = torch.tensor([0.1, 0.3, 0.4, 0.2], device=self.device)
            if X.shape[2] > 4:
                extra = torch.ones(X.shape[2] - 4, device=self.device) * 0.05
                weights = torch.cat([weights, extra])
        else:
            weights = torch.ones(X.shape[2], device=self.device) / X.shape[2]

        weights = weights / weights.sum()

        X_gray = torch.sum(X * weights.view(1, 1, -1), dim=2)

        X_gray = X_gray.unsqueeze(0).unsqueeze(0)
        I_PAN = I_PAN.unsqueeze(0).unsqueeze(0)

        ms_gradients_X = self.extract_multiscale_gradients(X_gray)
        ms_gradients_PAN = self.extract_multiscale_gradients(I_PAN)

        st_features_X = self.structure_tensor_features(X_gray)
        st_features_PAN = self.structure_tensor_features(I_PAN)

        gradient_loss = 0.0
        for gx, gp in zip(ms_gradients_X, ms_gradients_PAN):
            gx_mean = torch.mean(gx, dim=(2, 3), keepdim=True)
            gx_std = torch.std(gx, dim=(2, 3), keepdim=True) + 1e-6
            gp_mean = torch.mean(gp, dim=(2, 3), keepdim=True)
            gp_std = torch.std(gp, dim=(2, 3), keepdim=True) + 1e-6

            gx_norm = (gx - gx_mean) / gx_std
            gp_norm = (gp - gp_mean) / gp_std

            gradient_loss += torch.mean((gx_norm - gp_norm) ** 2)

        st_X_mean = torch.mean(st_features_X, dim=(2, 3), keepdim=True)
        st_X_std = torch.std(st_features_X, dim=(2, 3), keepdim=True) + 1e-6
        st_PAN_mean = torch.mean(st_features_PAN, dim=(2, 3), keepdim=True)
        st_PAN_std = torch.std(st_features_PAN, dim=(2, 3), keepdim=True) + 1e-6

        st_X_norm = (st_features_X - st_X_mean) / st_X_std
        st_PAN_norm = (st_features_PAN - st_PAN_mean) / st_PAN_std

        structure_loss = torch.mean((st_X_norm - st_PAN_norm) ** 2)

        kernel_size = 5
        padding = kernel_size // 2

        mean_filter = torch.ones((1, 1, kernel_size, kernel_size), device=self.device) / (kernel_size * kernel_size)
        local_mean_X = F.conv2d(X_gray, mean_filter, padding=padding)
        local_mean_PAN = F.conv2d(I_PAN, mean_filter, padding=padding)

        local_var_X = F.conv2d((X_gray - local_mean_X)**2, mean_filter, padding=padding)
        local_var_PAN = F.conv2d((I_PAN - local_mean_PAN)**2, mean_filter, padding=padding)

        var_X_mean = torch.mean(local_var_X)
        var_X_std = torch.std(local_var_X) + 1e-6
        var_PAN_mean = torch.mean(local_var_PAN)
        var_PAN_std = torch.std(local_var_PAN) + 1e-6

        var_X_norm = (local_var_X - var_X_mean) / var_X_std
        var_PAN_norm = (local_var_PAN - var_PAN_mean) / var_PAN_std

        texture_loss = torch.mean((var_X_norm - var_PAN_norm) ** 2)

        total_loss = lambda_gradient * gradient_loss + \
                    lambda_structure * structure_loss + \
                    lambda_texture * texture_loss

        return total_loss

    def compute_ergas_loss(self, X, I_PAN):
        a1 = torch.mean((X - I_PAN) ** 2, dim=(-2, -1))
        a2 = torch.mean(I_PAN, dim=(-2, -1)) ** 2
        com = a1 / a2
        ergas = 100 * (1 / self.ratio) * (com ** 0.5)

        return ergas.mean()

    def compute_spatial_fidelity_loss(self, X, I_LRMS, I_PAN, block_size, use_ergas=False, lamda=0.2):
        if X.dim() == 4:
            X = X.squeeze(0)

        if I_PAN.dim() == 3 and I_PAN.shape[0] == 1:
            I_PAN = I_PAN.squeeze(0)
        elif I_PAN.dim() == 4:
            I_PAN = I_PAN.squeeze(0).squeeze(0)

        if self.sensor == 'WV3' or self.sensor == 'WV2':
            if X.shape[2] >= 8:
                weights = torch.tensor([0.05, 0.05, 0.1, 0.2, 0.35, 0.15, 0.05, 0.05], device=self.device)
                if X.shape[2] > 8:
                    extra = torch.ones(X.shape[2] - 8, device=self.device) * 0.05
                    weights = torch.cat([weights, extra])
            else:
                weights = torch.ones(X.shape[2], device=self.device) / X.shape[2]
        elif self.sensor == 'QB':
            weights = torch.tensor([0.1, 0.3, 0.4, 0.2], device=self.device)
            if X.shape[2] > 4:
                extra = torch.ones(X.shape[2] - 4, device=self.device) * 0.05
                weights = torch.cat([weights, extra])
        else:
            weights = torch.ones(X.shape[2], device=self.device) / X.shape[2]

        weights = weights / weights.sum()

        X_gray = torch.sum(X * weights.view(1, 1, -1), dim=2)

        X_gray = X_gray.unsqueeze(0).unsqueeze(0)
        I_PAN = I_PAN.unsqueeze(0).unsqueeze(0)

        if use_ergas:
            l1_loss = torch.mean(torch.abs(X_gray - I_PAN))

            b, c, _, _ = X_gray.shape
            a1 = torch.mean((X_gray - I_PAN) ** 2, dim=(-2, -1))
            a2 = torch.mean(I_PAN, dim=(-2, -1)) ** 2
            com = a1 / (a2 + 1e-8)
            ergas = 100 * (1 / self.ratio) * torch.sqrt(com)
            ergas_loss = ergas.mean()

            total_loss = l1_loss + lamda * ergas_loss
        else:
            total_loss = torch.mean(torch.abs(X_gray - I_PAN))

        return total_loss

    def SDE_Loss(self, x, y):
            mse_loss = nn.MSELoss()
            loss = mse_loss(x, y)
            return loss


class SDE_Losses(nn.Module):
    def __init__(self, device):
        super(SDE_Losses, self).__init__()
        self.mse = nn.MSELoss().to(device)

    def forward(self, lms_rr, pan_rr):
        loss = self.mse(lms_rr, pan_rr)
        return loss


class pretrain_Losses(nn.Module):
    def __init__(self, device):
        super(pretrain_Losses, self).__init__()
        self.mse = nn.MSELoss().to(device)

    def forward(self, lms_rr, pan_rr):
        loss = self.mse(lms_rr, pan_rr)
        return loss
