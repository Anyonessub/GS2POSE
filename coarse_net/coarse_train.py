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

import cv2
from nets.deeplabv3_training import Focal_Loss, Dice_loss
import numpy as np
import matplotlib.pyplot as plt
from torch import unsqueeze
from utils.my_utils import MyData, unet, painting, compute_mIoU, per_class_iu, per_class_PA, dice_loss, \
    adaptive_upgrade_weight, initialize_weights
import time

# os.environ['CUDA_VISIBLE_DEVICES']='3'
def loss(out, target, classes, weight):
    # d_loss=Dice_loss(out[:,:-1,...], f.one_hot(target, classes).float())
    d_loss = dice_loss(f.softmax(out, dim=1).float(), f.one_hot(target, classes).permute(0, 3, 1, 2).float(),
                       multiclass=True)
    f_loss = Focal_Loss(out, target, cls_weights=weight, num_classes=classes, alpha=0.75)
    return f_loss + 0.3 * d_loss


def l1_loss_nonzero(y_true, y_pred):

    y_pred=y_pred.cpu().detach().numpy()
    y_true=y_true.cpu().detach().numpy()
    non_zero_mask = y_true != 0


    return np.sum(np.abs(y_true[non_zero_mask] - y_pred[non_zero_mask]))

class GetLoader(Dataset):
    def __init__(self, image_root, label_root, transform=None):
        self.image_root = image_root
        self.label_root = label_root
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_root) if os.path.isfile(os.path.join(image_root, f))]
        self.label_files = [f for f in os.listdir(label_root) if os.path.isfile(os.path.join(label_root, f))]
        self.image_files.sort()
        self.label_files.sort()


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

transform = transforms.Compose([
    transforms.ToTensor(),
])

def func():
    resume = 1

    if not os.path.exists('./split_dc_img'):
        os.mkdir('./split_dc_img')
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')
    classes = 3
    batch_size = 8
    start_epoch = 0
    num_epoch = 1000
    
    min_loss = 1.4

    cls_weights = torch.from_numpy(np.ones([classes], np.float32)).cuda()

    # 训练集
    train_image_root = 'data/000014/train/gt'
    train_label_root = 'data/000014/train/render'
    train_set = GetLoader(train_image_root, train_label_root, transform=transform)

    #验证集
    val_image_root = 'data/000014/val/gt'
    val_label_root = 'data/000014/val/render'
    val_set = GetLoader(val_image_root, val_label_root, transform=transform)


    G = unet(num_classes=classes, pretrained=False, backbone="resnet50").cuda()  # generator model

    G.apply(initialize_weights)


    g_optimizer = torch.optim.SGD(G.parameters(), lr=0.0005, momentum=0.9, weight_decay=0.0001)
    # g_optimizer = torch.optim.AdamW(G.parameters(), lr=0.001, weight_decay=0.0001)

    dataloader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4,
                            pin_memory=True)  # num_workers=4
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True, drop_last=True,num_workers=4,
                            pin_memory=True)

    train_steps = int((train_set.__len__()) / batch_size)
    val_steps = int((val_set.__len__()) / batch_size)

    train_loss = []
    val_loss = []

    L1_mean = nn.L1Loss(reduction='mean')

    if resume == 1:
        checkpoint_path = './checkpoints/checkpoint-27.pth'
        checkpoint = torch.load(checkpoint_path)
        G.load_state_dict(checkpoint["net"], strict=True)

    # train
    for epoch in range(start_epoch, num_epoch):
        losses = 0
        for step, data in enumerate(dataloader):  # data:(index,img,label,target)

            img = data[0].cuda()
            real_label = data[1].cuda()

            # img = img.cpu().numpy()

            fake_img,aux_image = G(img)  #640,480
            real_label = real_label.to(fake_img.dtype)

            # Create a mask for non-zero pixels in real_label
            mask = (real_label != 0)

            masked_fake_img = fake_img[mask]
            masked_real_label = real_label[mask]

            out_loss = L1_mean(masked_fake_img, masked_real_label)

            # out_loss = L1_mean(fake_img, real_label)
            g_loss = out_loss

            losses += g_loss
            g_loss.backward()
            g_optimizer.step()
            g_optimizer.zero_grad()
            if (step + 1) % 50 == 0:
                print('Epoch [{}/{}], Step [{}/{}],loss: {:.6f} '
                .format(
                    epoch + 1, num_epoch, step + 1, train_steps, 100 * g_loss.data
                ))

        train_loss.append((losses / (step + 1)).item())
        if (epoch + 1) % 1 == 0:
            G.eval()
            val_losses = 0
            # val_set.transform = True
            for step, data in enumerate(val_loader):
                label = data[1]
                val_label = label.cuda()
                val_img = data[0].cuda()
                with torch.no_grad():
                    fake_val,aux_val = G(val_img)

                # out_loss=criterion(fake_val,val_label)
                # aux_loss=criterion(aux_val,val_label)

                mask = (val_label != 0)

                masked_fake_val =  fake_val[mask]
                masked_val_label = val_label[mask]

                out_loss = L1_mean(masked_fake_val, masked_val_label)

                # out_loss = L1_mean(fake_val, val_label)
                # out_loss = l1_loss_nonzero(val_label, fake_val)
                v_loss = out_loss
                val_losses += v_loss


            val_loss.append((val_losses / (val_steps)).item())


            # print(str(epoch+1)+'avg-valloss:'+str(val_losses/(val_steps)),file=record)
            print("epoch-{} , val_loss={}".format(epoch + 1, val_losses / (val_steps)))
            
            com_loss = (val_losses / (val_steps)).cpu()
            com_loss = np.array(com_loss)

            # if com_loss<min_loss-0.001:
            #     checkpoint = {
            #         "net": G.state_dict(),
            #         # "optimizer": g_optimizer.state_dict(),
            #         # "epoch": epoch + 1,
            #     }
            #     torch.save(checkpoint, './checkpoints/checkpoint-{}.pth'.format(epoch + 1))
            #     min_loss = com_loss
            # G.train()
            # time.sleep(0.003)


    x1points = np.array(range(start_epoch + 1, num_epoch + 1, 1))
    y1points = np.array(train_loss) * 10
    y2points = np.array(val_loss) * 10
    plt.plot(x1points, y1points, c='b', label='train_loss')
    plt.plot(x1points, y2points, c='g', label='val_loss')
    plt.title('loss-epoch')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc=1)
    plt.savefig('./xception_result.jpg')
    plt.show()


if __name__ == "__main__":
    func()