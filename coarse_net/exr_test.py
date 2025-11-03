import os
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image
from torchvision.transforms import Resize
from PIL import Image
import torchvision.transforms.functional as F
# import OpenEXR
# import Imath

import cv2
from nets.deeplabv3_training import Focal_Loss, Dice_loss
import numpy as np
import matplotlib.pyplot as plt
from torch import unsqueeze
from utils.my_utils import MyData, unet, painting, compute_mIoU, per_class_iu, per_class_PA, dice_loss, \
    adaptive_upgrade_weight, initialize_weights
    
# os.environ['CUDA_VISIBLE_DEVICES']='3' 

"""
网络输出模型的NOCS图像
"""


def loss(out, target, classes, weight):
    # d_loss=Dice_loss(out[:,:-1,...], f.one_hot(target, classes).float())
    d_loss = dice_loss(f.softmax(out, dim=1).float(), f.one_hot(target, classes).permute(0, 3, 1, 2).float(),
                       multiclass=True)
    f_loss = Focal_Loss(out, target, cls_weights=weight, num_classes=classes, alpha=0.75)
    return f_loss + 0.3 * d_loss


def l1_loss_nonzero(y_true, y_pred):
    """
    y_true (numpy.ndarray): 实际值数组
    y_pred (numpy.ndarray): 预测值数组
    """
    # 找到y_true中不为0的部分
    y_pred=y_pred.cpu().detach().numpy()
    y_true=y_true.cpu().detach().numpy()
    non_zero_mask = y_true != 0

    # 只计算不为0部分的L1损失
    return np.sum(np.abs(y_true[non_zero_mask] - y_pred[non_zero_mask]))

class GetLoader(Dataset):
    def __init__(self, image_root, transform=None):
        self.image_root = image_root

        self.transform = transform
        self.image_files = [f for f in os.listdir(image_root) if os.path.isfile(os.path.join(image_root, f))]

        self.image_files.sort()

    def __getitem__(self, index):
        img_path = os.path.join(self.image_root, self.image_files[index])


        img = Image.open(img_path).convert("RGB")
        img = F.center_crop(img, 256)


        if self.transform:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.image_files)

# 定义图像和标签的转换
transform = transforms.Compose([
    transforms.ToTensor(),
])

def func():
    classes = 3
    batch_size = 1

    # 训练集
    test_image_root = 'lm_data/000014/gt'
    test_set = GetLoader(test_image_root, transform=transform)

    dataloader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4,
                            pin_memory=True)  # num_workers=4

    G = unet(num_classes=classes, pretrained=False, backbone="resnet50").cuda()  # generator model

    checkpoint_path = 'checkpoints/checkpoint-27.pth'
    checkpoint = torch.load(checkpoint_path)
    G.load_state_dict(checkpoint["net"], strict=True)
    G.eval()

    # train
    for step, data in enumerate(dataloader):
        img = data
        with torch.no_grad():
            fake_img, aux_image = G(img.cuda())

        # Get the corresponding filename from the dataset
        original_filename = test_set.image_files[step]
        original_basename, _ = os.path.splitext(original_filename)

        # Save the fake image with the same name as the input image
        save_path = f'lm_data/000014/output/{original_basename}.png'
        save_image(fake_img, save_path)

        print(f"Saved {save_path}")



if __name__ == "__main__":
    func()
