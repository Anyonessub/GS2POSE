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
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ORBFeatureMatchingLoss(nn.Module):
    def __init__(self, max_matches=50):
        super(ORBFeatureMatchingLoss, self).__init__()
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.max_matches = max_matches

    def forward(self, pred, gt):
        """
        pred: C x H x W CUDA Tensor
        gt: C x H x W CUDA Tensor
        """
        assert pred.dim() == 3
        assert gt.dim() == 3
        assert pred.size(1) == gt.size(1) and pred.size(2) == gt.size(2)

        C, H, W = pred.shape

        pred_np = pred.detach().cpu().permute(1, 2, 0).numpy()
        gt_np = gt.detach().cpu().permute(1, 2, 0).numpy()


        pred_np = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min() + 1e-8)
        pred_np = (pred_np * 255).astype(np.uint8)
        gt_np = (gt_np - gt_np.min()) / (gt_np.max() - gt_np.min() + 1e-8)
        gt_np = (gt_np * 255).astype(np.uint8)


        if pred_np.shape[2] == 3:
            pred_gray = cv2.cvtColor(pred_np, cv2.COLOR_BGR2GRAY)
        else:
            pred_gray = pred_np
        if gt_np.shape[2] == 3:
            gt_gray = cv2.cvtColor(gt_np, cv2.COLOR_BGR2GRAY)
        else:
            gt_gray = gt_np


        kp1, des1 = self.orb.detectAndCompute(pred_gray, None)
        kp2, des2 = self.orb.detectAndCompute(gt_gray, None)

        if des1 is None or des2 is None:

            return torch.tensor(1.0, device=pred.device, dtype=pred.dtype, requires_grad=True)


        matches = self.bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) == 0:
            return torch.tensor(1.0, device=pred.device, dtype=pred.dtype, requires_grad=True)

        N = min(self.max_matches, len(matches))
        matches = matches[:N]


        pred_pts = []
        gt_pts = []
        for m in matches:
            pt1 = kp1[m.queryIdx].pt  # (x, y)
            pt2 = kp2[m.trainIdx].pt
            pred_pts.append(pt1)
            gt_pts.append(pt2)

        pred_pts = np.array(pred_pts)
        gt_pts = np.array(gt_pts)

        pred_pts_tensor = torch.tensor(pred_pts, dtype=torch.float32, device=pred.device)
        gt_pts_tensor = torch.tensor(gt_pts, dtype=torch.float32, device=gt.device)

        # 归一化坐标
        pred_norm = pred_pts_tensor.clone()
        gt_norm = gt_pts_tensor.clone()

        pred_norm[:, 0] = (pred_norm[:, 0] / (W - 1)) * 2 - 1
        pred_norm[:, 1] = (pred_norm[:, 1] / (H - 1)) * 2 - 1
        gt_norm[:, 0] = (gt_norm[:, 0] / (W - 1)) * 2 - 1
        gt_norm[:, 1] = (gt_norm[:, 1] / (H - 1)) * 2 - 1

        pred_batch = pred.unsqueeze(0)  # 1 x C x H x W
        pred_grid = pred_norm.view(1, N, 1, 2)  # 1 x N x 1 x 2
        pred_sampled = F.grid_sample(pred_batch, pred_grid, align_corners=True)  # 1 x C x N x 1
        pred_sampled = pred_sampled.squeeze(3).squeeze(0).permute(1, 0)  # N x C

        gt_batch = gt.unsqueeze(0)  # 1 x C x H x W
        gt_grid = gt_norm.view(1, N, 1, 2)  # 1 x N x 1 x 2
        gt_sampled = F.grid_sample(gt_batch, gt_grid, align_corners=True)  # 1 x C x N x 1
        gt_sampled = gt_sampled.squeeze(3).squeeze(0).permute(1, 0)  # N x C

        distances = F.l1_loss(pred_sampled, gt_sampled, reduction='none')  # N x C
        distances = distances.mean(dim=1)  # N

        loss = distances.mean()

        return loss

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
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    opacity_mask = (opacity > 0.99).view(*mask_shape)#足够不透明

    l1 = torch.abs(image*opacity_mask - gt_image) # 逐像素求绝对差
    loss = 0.7 * l1.mean() + 0.3 * (1.0 - ssim(image , gt_image))
    return loss

def get_loss_tracking_rgbd(viewpoint, image, opacity, gt_image,depth,gt_depth,gt_mask):
    image = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    opacity_mask = (opacity > 0.99).view(*mask_shape)
    l1_depth = torch.abs(depth*gt_mask - gt_depth*gt_mask)
    l1_depth = l1_depth.mean()

    l1 = opacity_mask*torch.abs(image*opacity_mask - gt_image*opacity_mask)

    loss = 0.3 * l1.mean() + 0.7 * (1.0 - ssim(image, gt_image)) + 0.00002 * l1_depth

    return loss


def get_loss_tracking_rgbd_2(viewpoint, image, opacity, gt_image,depth,gt_depth,gt_mask):
    image = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    opacity_mask = (opacity > 0.99).view(*mask_shape)
    l1_depth = torch.abs(depth*gt_mask - gt_depth*gt_mask)
    l1_depth = l1_depth.mean()
    # print(l1_depth)
    l1 = opacity*torch.abs(image*opacity_mask - gt_image)

    loss = 0.2 * l1.mean() + 0.8 * (1.0 - ssim(image , gt_image))+0.00004*l1_depth
    # print(f" loss l1_depth {loss} {l1_depth*0.00001}")
    return loss

def custom_loss(image, opacity_mask, gt_image, threshold=0.07):
    predicted_image = image * opacity_mask

    abs_diff = torch.abs(predicted_image - gt_image)

    mask = (abs_diff < threshold).float()

    loss = (abs_diff * mask).sum() / mask.sum()

    return loss


def get_loss_tracking_rgb_1(viewpoint, image, gt_mask ,opacity, gt_image):
    image = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    opacity_mask = (opacity > 0.99).view(*mask_shape)

    l1 = opacity * torch.abs(image * opacity_mask - gt_image)
    ori_loss = 0.7 * l1.mean() + 0.3 * (1.0 - ssim(image * opacity_mask, gt_image))
    return ori_loss

def get_loss_tracking_rgb_2(viewpoint, image, gt_mask ,opacity, gt_image):
    image = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    opacity_mask = (opacity > 0.99).view(*mask_shape)
    # l1 = custom_loss(image, opacity_mask, gt_image, threshold=0.1)
    l1 = opacity * torch.abs(image * opacity_mask - gt_image)
    ori_loss = 0.7 * l1.mean() + 0.3 * (1.0 - ssim(image, gt_image))
    return ori_loss

def get_loss_tracking_rgb_3(viewpoint, image, opacity, gt_image):
    image = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = 0.1
    opacity_mask = (opacity > 0.99).view(*mask_shape)
    l1 = opacity * torch.abs(image * opacity_mask - gt_image)
    loss = l1.mean()
    return loss

def fix_z_orb(image1, image2, camera_matrix):
    # 将图片转换为灰度
    gray1 = cv2.cvtColor(image1.detach().cpu().permute(1, 2, 0).numpy()*255, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2.detach().cpu().permute(1, 2, 0).numpy()*255, cv2.COLOR_BGR2GRAY)

    gray1 = gray1.astype(np.uint8)
    gray2 = gray2.astype(np.uint8)

    orb = cv2.ORB_create()

    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(descriptors1, descriptors2)

    matches = sorted(matches, key=lambda x: x.distance)

    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

    if H is not None:

        fx = camera_matrix.fx
        fy = camera_matrix.fy

        scale = (H[0, 0] + H[1, 1]) / 2
        distance_difference = (fx * fy) / scale

        return distance_difference
    else:
        return None