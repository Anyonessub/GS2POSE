import torch
import os
from torch.utils.data import DataLoader, Dataset

from torchvision.utils import save_image
from torchvision.transforms import Resize
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

from utils.my_utils import MyData, unet, painting, compute_mIoU, per_class_iu, per_class_PA
from torchvision import transforms
import torchvision.transforms.functional as F


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


save_dir = 'F:/6D/GS_seg/split_test_img/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

class GetLoader(Dataset):
    def __init__(self, image_root, label_root, transform=None):
        self.image_root = image_root
        self.label_root = label_root
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_root) if os.path.isfile(os.path.join(image_root, f))]
        self.label_files = [f for f in os.listdir(label_root) if os.path.isfile(os.path.join(label_root, f))]
        self.image_files.sort()
        self.label_files.sort()
        assert len(self.image_files) == len(self.label_files), "图像和标签的数量不匹配"

    def __getitem__(self, index):
        img_path = os.path.join(self.image_root, self.image_files[index])
        label_path = os.path.join(self.label_root, self.label_files[index])

        img = Image.open(img_path).convert("RGB")
        label = Image.open(label_path)
        crop_height, crop_width = 256, 256
        img_width, img_height = img.size
        img = img.crop((
        (img_width - crop_width) // 2,
        (img_height - crop_height) // 2,
        (img_width + crop_width) // 2,
        (img_height + crop_height) // 2
    ))
        label = label.crop((
        (img_width - crop_width) // 2,
        (img_height - crop_height) // 2,
        (img_width + crop_width) // 2,
        (img_height + crop_height) // 2
    ))

        if self.transform:
            img = self.transform(img)
            label = self.transform(label)

        return img, label

    def __len__(self):
        return len(self.image_files)

# 定义图像和标签的转换
transform = transforms.Compose([
    transforms.ToTensor(),
])


def func():
    classes = 3
    batch_size = 1
    cls_weights = torch.from_numpy(np.ones([classes], np.float32)).cuda()

    val_image_root = 'F:/6D/unet2/val/rgb/'
    val_label_root = 'F:/6D/unet2/val/render/'
    val_set = GetLoader(val_image_root, val_label_root, transform=transform)

    dataloader_target = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4,
                            pin_memory=True)

    totalsteps = int((val_set.__len__()) / batch_size)
    G = unet(num_classes=classes, pretrained=False, backbone="resnet50").cuda()  # generator model
    for param in G.parameters():
        param.requires_grad = False


    checkpoint_path = 'F:/6D/GS_seg/checkpoints/checkpoint-1000.pth'
    checkpoint = torch.load(checkpoint_path)
    G.load_state_dict(checkpoint["net"], strict=True)
    G.eval()

    # test
    loss = []
    for step, data in enumerate(dataloader_target):  # data:(index,img,label,target)
        img = data[0]
        real_label = data[1].cuda()

        with torch.no_grad():
            fake_img,aux_image= G(img.cuda())

        fake_img=fake_img.cpu().numpy()

        fake_img = torch.from_numpy(fake_img)

        if (step + 1) % 1 == 0:
            # fake_img.save('./split_test_img/{}-fake_images.png'.format(step + 1))
            save_image(fake_img, './split_test_img/{}-fake_images.png'.format(step + 1))
            save_image(img, './split_test_img/{}-raw_images.png'.format(step + 1))
            save_image(real_label, './split_test_img/{}-real_label.png'.format(step + 1))



if __name__ == "__main__":
    func()