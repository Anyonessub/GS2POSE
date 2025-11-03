import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image
from torchvision.transforms import Resize
import torchvision.transforms.functional as F
from PIL import Image
import os
import cv2
from nets.deeplabv3_plus import DeepLab
from nets.deeplabv3_training import CE_Loss
from nets.unet import Unet
import numpy as np
import matplotlib.pyplot as plt
from torch import unsqueeze
import random


def image_Splicing(img_1, img_2, flag='x'):
    size1, size2 = img_1.size, img_2.size
    if flag == 'x':
        joint = Image.new("RGB", (size1[0] + size2[0], size1[1]))
        loc1, loc2 = (0, 0), (size1[0], 0)
    else:
        joint = Image.new("RGB", (size1[0], size2[1]+size1[1]))
        loc1, loc2 = (0, 0), (0, size1[1])
    joint.paste(img_1, loc1)
    joint.paste(img_2, loc2)
    return joint

from torch import Tensor

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]
def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]-1):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]
def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.)
        m.bias.data.fill_(1e-4)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0.0, 0.0001)
        m.bias.data.zero_()


# class FocalLoss(nn.Module):
#     r"""
#         This criterion is a implemenation of Focal Loss, which is proposed in 
#         Focal Loss for Dense Object Detection.
#             Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
#         The losses are averaged across observations for each minibatch.
#         Args:
#             alpha(1D Tensor, Variable) : the scalar factor for this criterion
#             gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
#                                    putting more focus on hard, misclassiﬁed examples
#             size_average(bool): By default, the losses are averaged over observations for each minibatch.
#                                 However, if the field size_average is set to False, the losses are
#                                 instead summed for each minibatch.
#     """
#     def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
#         super(FocalLoss, self).__init__()
#         if alpha is None:
#             self.alpha = Variable(torch.ones(class_num, 1))
#         else:
#             if isinstance(alpha, Variable):
#                 self.alpha = alpha
#             else:
#                 self.alpha = Variable(alpha)
#         self.gamma = gamma
#         self.class_num = class_num
#         self.size_average = size_average
 
#     def forward(self, inputs, targets):
#         N = inputs.size(0)
#         C = inputs.size(1)
#         P = f.softmax(inputs,dim=1)
 
#         class_mask = inputs.data.new(N, C).fill_(0)
#         class_mask = Variable(class_mask)
#         ids = targets.view(-1, 1)
#         class_mask.scatter_(1, ids.data, 1.)
#         #print(class_mask)
 
 
#         if inputs.is_cuda and not self.alpha.is_cuda:
#             self.alpha = self.alpha.cuda()
#         alpha = self.alpha[ids.data.view(-1)]
 
#         probs = (P*class_mask).sum(1).view(-1,1)
 
#         log_p = probs.log()
#         #print('probs size= {}'.format(probs.size()))
#         #print(probs)
 
#         batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
#         #print('-----bacth_loss------')
#         #print(batch_loss)
 
 
#         if self.size_average:
#             loss = batch_loss.mean()
#         else:
#             loss = batch_loss.sum()
#         return loss
"""
# 多分类的 FocalLoss
如果是二分类问题，alpha 可以设置为一个值
如果是多分类问题，这里只能取list 外面要设置好list 并且长度要与分类bin大小一致,并且alpha的和要为1  
比如dist 的alpha=[0.02777]*36 +[0.00028] 这里是37个分类，设置前36个分类系数一样，最后一个分类权重系数小于前面的。

注意: 这里默认 input是没有经过softmax的，并且把shape 由H*W 2D转成1D的，然后再计算
"""

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = f.log_softmax(input,dim=1) # 这里转成log(pt)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()



class MyData(Dataset):  # 继承Dataset
    def __init__(self, origin_dir, label_dir, target_dir, transform=True):  # __init__是初始化该类的一些基础参数
        self.origin_dir = origin_dir  # 文件目录
        self.label_dir=label_dir
        self.target_dir=target_dir
        self.transform=transform
        self.transform_img =transforms.Compose([
                        transforms.Resize((256,256),interpolation=transforms.InterpolationMode.BILINEAR),
                        #transforms.ColorJitter(brightness=0.2),
                        transforms.ToTensor(),
                        ])  # 图像变换
        self.transform_label = transforms.Resize((256,256),interpolation=transforms.InterpolationMode.NEAREST)# 标签变换,!!!需要大于等于3维!!!
        self.images = sorted(os.listdir(self.origin_dir))  # 目录里的所有文件
    def __len__(self):  # 返回整个数据集的大小
        return len(self.images)

    def __getpath__(self,indexs,file):
        for index in indexs:
            print(self.images[index],file=file)
    
    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        #读取图片
        image_index = self.images[index][:-4]  # 根据索引index获取该图片
        img_path = os.path.join(self.origin_dir, image_index)  # 获取索引为index的图片的路径名
        img = Image.open(img_path+'.jpg').convert('RGB')  # 读取该图片
        #读取标签
        label_path = os.path.join(self.label_dir, image_index)
        label=torch.load(label_path+'.pt')
        #读取真值
        target_path = os.path.join(self.target_dir, image_index)
        target = Image.open(target_path+'.png').convert('RGB')
        #变换
        if self.transform==True:
            p1=random.randint(0,1)
            p2=random.randint(0,1)
            h,w=256,256
            i=random.randint(0,img.size[-1]-h-1)#高
            j=random.randint(0,img.size[-2]-w-1)#宽
            img=F.crop(img,i,j,h,w)
            target=F.crop(target,i,j,h,w)
            label=F.crop(label,i,j,h,w)
            
            if p1==1:
                img=F.hflip(img)
                target=F.hflip(target)
                label=F.hflip(label)
            if p2==1:
                img=F.vflip(img)
                target=F.vflip(target)
                label=F.vflip(label)
            
        img = self.transform_img(img).float()
            
        label = torch.unsqueeze(label,dim=0)
        label = self.transform_label(label)[0].long()
            
        target = self.transform_label(target)
        target = transforms.ToTensor()(target).float()
        return (index,img,label,target)  # 返回该样本和索引

from torch.utils.data import Subset
class CustomSubset(Subset):
    '''A custom subset class'''
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.targets = dataset.targets # 保留targets属性
        self.classes = dataset.classes # 保留classes属性

    def __getitem__(self, idx): #同时支持索引访问操作
        x, y = self.dataset[self.indices[idx]]
        return x, y

    def __len__(self): # 同时支持取长度操作
        return len(self.indices)

class Mylabel(Dataset):  # 继承Dataset
    def __init__(self, root_dir, transform=None):  # __init__是初始化该类的一些基础参数
        self.root_dir = root_dir  # 文件目录
        self.labels = sorted(os.listdir(self.root_dir))  # 目录里的所有文件

    def __len__(self):  # 返回整个数据集的大小
        return len(self.labels)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        label_index = self.labels[index]  # 根据索引index获取该图片
        label_path = os.path.join(self.root_dir, label_index)  # 获取索引为index的图片的路径名
        label=torch.load(label_path)
        return label # 返回该样本和路径

class discriminator(nn.Module):
    def __init__(self,num_classes,h_in,w_in, ndf=64):
        super().__init__()
        self.conv1=nn.Sequential(
                nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1),
        )
        self.fc=nn.Sequential(
                nn.Flatten(),
                nn.Linear((h_in/32)*(w_in/32),1)
        )
    def forward(self, x):
        '''
        x: batch, channel, width, height
        '''
        x = self.conv1(x)
        return x


class generator(nn.Module):
    def __init__(self, num_classes,backbone,pretrained=True,downsample_factor=16):
        super().__init__()
        self.normalize=transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.generate=DeepLab(num_classes=num_classes,backbone=backbone,pretrained=pretrained,downsample_factor=downsample_factor)
    def forward(self, x):
        x = self.normalize(x)
        x = self.generate.forward(x)
        return x

class unet(nn.Module):
    def __init__(self, num_classes,backbone,pretrained=True,downsample_factor=16):
        super().__init__()
        self.normalize=transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.generate=Unet(num_classes=num_classes,backbone=backbone,pretrained=pretrained)
    def forward(self, x):
        x = self.normalize(x) #Imagenet上计算得到的均值与标准差，便于加快模型收敛
        x,aux = self.generate.forward(x)
        return x,aux
class painting():
    def __init__(self,batchnum,channel,h,w):
        self.colors=[(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128),(0,128,128),(128,128,128),(192,0,0),(64,0,0),(0,0,0)]#rgb
        self.batchnum=batchnum
        self.channel=channel
        self.height=h
        self.width=w
    def paint(self,imgs):
        results=[]
        for i in range(self.batchnum):
            result=np.zeros((self.height, self.width, 3))#h,w,c
            img=imgs[i]
            img=torch.argmax(img,dim=0).cpu().numpy()
            for c in range(self.channel):
                result[:, :, 0] += ((img[:, :] == c ) * self.colors[c][0]).astype('uint8')
                result[:, :, 1] += ((img[:, :] == c ) * self.colors[c][1]).astype('uint8')
                result[:, :, 2] += ((img[:, :] == c ) * self.colors[c][2]).astype('uint8')
            results.append(np.uint8(result))
        return np.hstack(results)

class classify():
    def classifier(self,img,batchnum,channel,high,wide,classes=9):#输入batch_size，h，w，c 输出batch_size，h，w tensor
        result=np.zeros((batchnum,high,wide))
        color={(128,0,0):0,(0,128,0):1,(128,128,0):2,(0,0,128):3,(128,0,128):4,(0,128,128):5,(128,128,128):6,(192,0,0):7,(64,0,0):8,(0,0,0):9}
        for n in range(batchnum):
            single_img=img[n]
            # input_tensor = single_img.clone().detach().to(torch.device('cpu'))# 到cpu
            # single_img = input_tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            for h in range(high):
                for w in range(wide):
                    g=tuple(single_img[h,w,:])
                    c=color.get(g,None)
                    result[n][h][w]=c
        return torch.from_numpy(result).long()
    def reverse_classifier(self,img,batchnum,channel,h,w):
        img=img.cpu().numpy()
        result=torch.zeros(batchnum,channel,h,w)
        color={(128,0,0):3,(0,128,0):4,(128,128,0):5,(0,0,128):0,(128,0,128):4,(0,128,128):5,(128,128,128):6,(192,0,0):7,(64,0,0):8,(0,0,0):9}
        for n in range(batchnum):
            single_img=img[n]
            for h in range(h):
                for w in range(w):
                    result[n][color.get(single_img[:,h,w],None)][h][w]=1
        return result

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target
class checkpoint():
    def __init__(self,net,optimizer,epoch):
        self.net=net,
        self.optimizer=optimizer,
        self.epoch=epoch
    def save_checkpoint(self):
        if not os.path.isdir("./checkpoints"):
            os.mkdir("./checkpoints")
        torch.save(checkpoint, './models/checkpoint/ckpt_best_%s.pth' %(str(self.epoch)))

def mean_iou(input, target, classes = 2):
    """  compute the value of mean iou
    :param input:  2d array, int, prediction
    :param target: 2d array, int, ground truth
    :param classes: int, the number of class
    :return:
        miou: float, the value of miou
    """
    miou = 0
    real_classes=0
    for i in range(classes):
        intersection = np.logical_and(target == i, input == i)
        # print(intersection.any())
        union = np.logical_or(target == i, input == i)
        if np.sum(union):
            temp = np.sum(intersection) / np.sum(union)
            real_classes+=1
        else:temp=0
        miou += temp
    return  miou/real_classes
# 设标签宽W，长H
def fast_hist(a, b, n):
    #--------------------------------------------------------------------------------#
    #   a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的预测结果，形状(H×W,)
    #--------------------------------------------------------------------------------#
    k = (a >= 0) & (a < n)
    #--------------------------------------------------------------------------------#
    #   np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    #   返回中，写对角线上的为分类正确的像素点
    #--------------------------------------------------------------------------------#
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)  

def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1) 

def per_class_PA(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1) 

def compute_mIoU(hist,gt_img, pred_img, num_classes, name_classes,step,steps,file=None,fileout=True):  

    #------------------------------------------------#
    #   对一张图片计算classes×classes的hist矩阵，并累加
    #------------------------------------------------#
    hist += fast_hist(gt_img.flatten(), pred_img.flatten(),num_classes)  
      
    if (step+1) % 100 == 0:
        if fileout==False:
            print('{:d} / {:d}: mIou-{:0.2f}; mPA-{:0.2f}'.format(step+1, steps,
                                                    100 * np.nanmean(per_class_iu(hist)),
                                                    100 * np.nanmean(per_class_PA(hist))))  
        else:
            print('{:d} / {:d}: mIou-{:0.2f}; mPA-{:0.2f}'.format(step+1, steps,
                                                    100 * np.nanmean(per_class_iu(hist)),
                                                    100 * np.nanmean(per_class_PA(hist))),file=file)  
    return hist

def scale_nearest(mat,Weight_out,Height_out):
    #获取图像的大小
    Height_in,Weight_in=mat.shape
    #创建输出图像
    outmat=np.zeros((Height_out,Weight_out))
 
    for x in range(Height_out):
        for y in range(Weight_out):
            #计算输出图像坐标（i,j）坐标使用输入图像中的哪个坐标来填充
            x_out=round(x*(Height_in/Height_out))
            y_out=round(y*(Weight_in/Weight_out))
            #插值
            outmat[x,y]=mat[x_out,y_out]
    return outmat
''' 
example:
    tensor1=torch.load('E:\Mars-seg data set\MSL\SegmentationClasstensor/121_0121ML0007580000103880E01_DXXX.pt')
    numpy1=tensor1.numpy()
    numpy2=scale_nearest(numpy1,256,256)
    paint=painting(batchnum=1,channel=10,h=256,w=256)
    tensor2=torch.from_numpy(numpy2).long()
    fake_img =Image.fromarray(paint.paint(torch.squeeze(tensor2,dim=0)))
    fake_img.save('test.png')
'''

def adaptive_upgrade_weight(weight,hist,dim,background_remove=False,offset=0.5):
    mious=per_class_iu(hist)
    new_weight=torch.reciprocal(torch.from_numpy(mious+offset))
    if background_remove==True:
        new_weight[dim]=0
    return new_weight.float()



