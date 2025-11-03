#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from math import exp

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

def get_loss_mapping_rgb(image, viewpoint):
    gt_image = viewpoint.gt_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = 0.01

    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)

    return l1_rgb.mean()

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l1_loss_weight(network_output, gt):
    image = gt.detach().cpu().numpy().transpose((1, 2, 0))
    rgb_raw_gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    sobelx = cv2.Sobel(rgb_raw_gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(rgb_raw_gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel_merge = np.sqrt(sobelx * sobelx + sobely * sobely) + 1e-10
    sobel_merge = np.exp(sobel_merge)
    sobel_merge /= np.max(sobel_merge)
    sobel_merge = torch.from_numpy(sobel_merge)[None, ...].to(gt.device)

    return torch.abs((network_output - gt) * sobel_merge).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def iou_loss(pred_mask, gt_mask, smooth=1e-6):
    pred_mask = pred_mask.view(-1)
    gt_mask = gt_mask.view(-1)

    intersection = (pred_mask * gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum() - intersection

    return 1 - (intersection + smooth) / (union + smooth)  # 返回1 - IoU作为损失

def get_loss_tracking_rgb(viewpoint, image, opacity, gt_image):
    image = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    l1 = opacity * torch.abs(image - gt_image) # 逐像素求绝对差
    loss = 0.3 * l1.mean() + 0.7 * (1.0 - ssim(image , gt_image))
    return loss

def get_loss_tracking_rgbd(
    image, gt_image ,depth, gt_depth, opacity,viewpoint,gt_mask
):
    alpha = 0.95
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)#不能太远
    opacity_mask = (opacity > 0.95).view(*depth.shape)#足够不透明

    l1_rgb = get_loss_tracking_rgb_1(viewpoint,image, opacity, gt_mask,gt_image)
    depth_mask = depth_pixel_mask * opacity_mask
    l1_depth = torch.abs(depth * depth_mask - gt_depth * depth_mask)
    return alpha * l1_rgb + (1 - alpha) * l1_depth.mean()

def custom_loss(image, opacity_mask, gt_image, threshold=0.07):
    # 计算预测图像
    predicted_image = image * opacity_mask

    # 计算绝对差异
    abs_diff = torch.abs(predicted_image - gt_image)

    # 创建一个掩码，标记哪些像素的颜色差异小于阈值
    mask = (abs_diff < threshold).float()

    # 计算加权损失，只对颜色差异小于阈值的像素计算损失
    loss = (abs_diff * mask).sum() / mask.sum()  # 归一化以避免因掩码导致的损失为0的情况

    return loss
def get_loss_tracking_rgb_1(viewpoint, image, opacity, gt_mask ,gt_image):
    image = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    opacity_mask = (opacity > 0.99).view(*mask_shape)#足够不透明
    l1 = opacity * torch.abs(image * opacity_mask * gt_mask - gt_image * gt_mask)  # 逐像素求绝对差
    # l1 = custom_loss(image, opacity_mask, gt_image, threshold=0.1)
    loss = 0.5 * l1.mean() + 0.5 * (1.0 - ssim(image * opacity_mask * gt_mask, gt_image * gt_mask))

    return loss # + mask_loss

def get_loss_tracking_rgb_2(viewpoint, image, gt_mask ,opacity, gt_image):
    image = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    opacity_mask = (opacity > 0.99).view(*mask_shape)#足够不透明
    # l1 = custom_loss(image, opacity_mask, gt_image, threshold=0.1)
    l1 = opacity * torch.abs(image * opacity_mask * gt_mask - gt_image * gt_mask)  # 逐像素求绝对差
    loss = 0.5 * l1.mean() + 0.5 * (1.0 - ssim(image * opacity_mask * gt_mask, gt_image * gt_mask))
    return loss

def get_loss_tracking_rgb_3(viewpoint, image, opacity, gt_image):
    image = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = 0.1
    opacity_mask = (opacity > 0.99).view(*mask_shape)#足够不透明
    l1 = opacity * torch.abs(image * opacity_mask - gt_image) # 逐像素求绝对差
    loss = l1.mean()
    return loss