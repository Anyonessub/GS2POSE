import os
# os.environ['CUDA_VISIBLE_DEVICES']='0'
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

import cv2
from nets.deeplabv3_training import Focal_Loss, Dice_loss
import numpy as np
import matplotlib.pyplot as plt
from torch import unsqueeze
from utils.my_utils import MyData, unet, painting, compute_mIoU, per_class_iu, per_class_PA, dice_loss, \
    adaptive_upgrade_weight, initialize_weights


def loss(out, target, classes, weight):
    # d_loss=Dice_loss(out[:,:-1,...], f.one_hot(target, classes).float())
    d_loss = dice_loss(f.softmax(out, dim=1).float(), f.one_hot(target, classes).permute(0, 3, 1, 2).float(),
                       multiclass=True)
    f_loss = Focal_Loss(out, target, cls_weights=weight, num_classes=classes, alpha=0.75)
    return f_loss + 0.3 * d_loss

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
    resume = 0

    if not os.path.exists('./split_dc_img'):
        os.mkdir('./split_dc_img')
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')
    classes = 10
    batch_size = 2
    num_epoch = 1200
    start_epoch = 0
    name_classes = ['Martian soil', 'Sands', 'Gravel', 'Bedrock', 'Rocks', 'Tracks', 'Shadows', 'Background', 'Unknown',
                    '__BACKGROUND__']
    cls_weights = torch.from_numpy(np.ones([classes], np.float32)).cuda()

    # 训练集
    train_image_root = 'E:/NOCS/unet2/data/train/rgb/'
    train_label_root = 'E:/NOCS/unet2/data/train/nocs/'
    train_set = GetLoader(train_image_root, train_label_root, transform=transform)

    #验证集
    val_image_root = 'E:/NOCS/unet2/data/val/rgb/'
    val_label_root = 'E:/NOCS/unet2/data/val/nocs/'
    val_set = GetLoader(val_image_root, val_label_root, transform=transform)

    # with open("./log_train.txt", "r", encoding="utf-8") as file:
    #     train_index = file.read().splitlines()
    # with open("./log_val.txt", "r", encoding="utf-8") as file:
    #     val_index = file.read().splitlines()
    #
    #
    # train_set = MyData(
    #     origin_dir='E:/mars/MER/JPEGImages/',
    #     label_dir='E:/mars/MER/SegmentationClasstensor',
    #     target_dir='E:/mars/MER/SegmentationClassPNG/', eval_index=train_index
    # )
    # val_set = MyData(
    #     origin_dir='E:/mars/MER/JPEGImages/',
    #     label_dir='E:/mars/MER/SegmentationClasstensor',
    #     target_dir='E:/mars/MER/SegmentationClassPNG/', eval_index=val_index
    # )


    G = unet(num_classes=classes, pretrained=False, backbone="resnet50").cuda()  # generator model

    G.apply(initialize_weights)


    g_optimizer = torch.optim.SGD(G.parameters(), lr=0.0005, momentum=0.9, weight_decay=0.0001)

    # paint=painting(batchnum=batch_size,channel=classes,h=256,w=256)

    dataloader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4,
                            pin_memory=True)  # num_workers=4
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True, drop_last=True)

    train_steps = int((train_set.__len__()) / batch_size)
    val_steps = int((val_set.__len__()) / batch_size)

    train_loss = []
    val_loss = []
    miou_total = []
    max_miou = 0
    if resume == 1:
        checkpoint_path = './checkpoints/checkpoint-435.pth'
        checkpoint = torch.load(checkpoint_path)
        G.load_state_dict(checkpoint["net"], strict=True)
        # start_epoch=checkpoint["epoch"]
        # g_optimizer.load_state_dict(checkpoint["optimizer"])
        max_miou = checkpoint["miou"]

    # train
    for epoch in range(start_epoch, num_epoch):
        losses = 0
        for step, data in enumerate(dataloader):  # data:(index,img,label,target)

            img = data[0].cuda()
            real_label = data[1].cuda()

            fake_img = G(img)

            out_loss = loss(fake_img, real_label, classes, cls_weights)
            # aux_loss = loss(aux_img, real_label, classes, cls_weights)
            g_loss = 0.6 * out_loss

            losses += g_loss
            g_loss.backward()
            g_optimizer.step()
            g_optimizer.zero_grad()
            if (step + 1) % 50 == 0:
                # print('Epoch [{}/{}], Step [{}/{}],loss: {:.6f} '
                #     .format(
                #     epoch+1, num_epoch,step+1,train_steps, 100*g_loss.data
                #     ),file=record)
                print('Epoch [{}/{}], Step [{}/{}],loss: {:.6f} '
                .format(
                    epoch + 1, num_epoch, step + 1, train_steps, 100 * g_loss.data
                ))
            # if (step+1) % 200 and (epoch+1)%25 == 0:
            #     fake_img =Image.fromarray(paint.paint(fake_img))
            #     fake_img.save('./split_dc_img/fake_images-{}.png'.format(epoch+1))
            #     save_image(img, './split_dc_img/raw_images-{}.png'.format(epoch+1))
            #     target=data[3]
            #     save_image(target, './split_dc_img/real_images-{}.png'.format(epoch+1))

        train_loss.append((losses / (step + 1)).item())
        if (epoch + 1) % 1 == 0:
            G.eval()
            val_losses = 0
            val_set.transform = False
            for step, data in enumerate(val_loader):
                label = data[2]
                val_label = label.cuda()
                val_img = data[1].cuda()
                with torch.no_grad():
                    fake_val = G(val_img)

                # out_loss=criterion(fake_val,val_label)
                # aux_loss=criterion(aux_val,val_label)
                out_loss = loss(fake_val, val_label, classes, cls_weights)
                v_loss = 0.6 * out_loss
                val_losses += v_loss

                if step == 0: hist = np.zeros((classes, classes))
                for j in range(batch_size):
                    fake = torch.argmax(fake_val[j], dim=0).cpu().numpy()
                    real = label[j].numpy()
                    hist, mious = compute_mIoU(hist=hist, pred_img=fake, gt_img=real, num_classes=classes, step=step,
                                               steps=val_steps)

            train_set.transform = True
            miou = 100 * np.nanmean(mious[:-1])
            val_loss.append((val_losses / (val_steps)).item())
            miou_total.append(miou)
            cls_weights = adaptive_upgrade_weight(mious, dim=9, offset=0.5, background_remove=True).cuda()
            # print(str(epoch+1)+'avg-valloss:'+str(val_losses/(val_steps)),file=record)
            print("epoch-{} , val_loss={} , miou={}".format(epoch + 1, val_losses / (val_steps), miou))
            if miou >= (max_miou - 0.2) and miou >= 30:
                max_miou = miou
                checkpoint = {
                    "net": G.state_dict(),
                    # "optimizer": g_optimizer.state_dict(),
                    # "epoch": epoch + 1,
                    "miou": miou
                }
                torch.save(checkpoint, './checkpoints/checkpoint-{}.pth'.format(epoch + 1))
            G.train()

    mIoUs = per_class_iu(hist)[:-1]
    mPA = per_class_PA(hist)[:-1]

    for ind_class in range(classes - 1):
        print('===>' + name_classes[ind_class] + ':\tmIou-' + str(round(mIoUs[ind_class] * 100, 2)) + '; mPA-' + str(
            round(mPA[ind_class] * 100, 2)))

    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)) + '; mPA: ' + str(round(np.nanmean(mPA) * 100, 2)))
    x1points = np.array(range(start_epoch + 1, num_epoch + 1, 1))
    y1points = np.array(train_loss) * 10
    y2points = np.array(val_loss) * 10
    y3points = np.array(miou_total)
    plt.plot(x1points, y1points, c='b', label='train_loss')
    plt.plot(x1points, y2points, c='g', label='val_loss')
    plt.plot(x1points, y3points, c='r', label='val_miou')
    plt.title('loss-epoch')
    plt.xlabel('epoch')
    plt.ylabel('loss\miou')
    plt.legend(loc=1)
    plt.savefig('./xception_result.jpg')
    plt.show()


if __name__ == "__main__":
    func()