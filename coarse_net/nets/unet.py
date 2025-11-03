import torch
import torch.nn as nn

from nets.resnet import resnet50
from nets.vgg import VGG16
from utils.polarized_selfattention import ParallelPolarizedSelfAttention
from utils.stripepool import StripPooling
import torch.nn.functional as F

#反卷积操作



class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        # self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.up = UpsampleModel(in_channel=in_size,out_channel=in_size)
        self.relu   = nn.ReLU(inplace = True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs


class mini_ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=2, bn_mom=0.1):
        super(mini_ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, bias=False),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=1 * rate, padding_mode='replicate', dilation=rate, bias=False),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=2 * rate, padding_mode='replicate', dilation=2 * rate, bias=False),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 3, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.psa = ParallelPolarizedSelfAttention(dim_out)

    def forward(self, x):
        # -----------------------------------------#
        #   一共三个分支
        # -----------------------------------------#
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)

        # -----------------------------------------#
        #   将三个分支的内容堆叠起来
        #   然后1x1卷积整合特征。
        # -----------------------------------------#
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2], dim=1)
        result = self.conv_cat(feature_cat)
        return self.psa(result)
    
    
class _PSPModule(nn.Module):
    def __init__(self, in_channels, pool_sizes, norm_layer):
        super(_PSPModule, self).__init__()
        out_channels = in_channels // len(pool_sizes)
        #-----------------------------------------------------#
        #   分区域进行平均池化
        #-----------------------------------------------------#
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, pool_size, norm_layer) for pool_size in pool_sizes])
        
        # self.bottleneck = nn.Sequential(
        #     nn.Conv2d(in_channels + (out_channels * len(pool_sizes)), out_channels, kernel_size=3, padding=1, bias=False),
        #     norm_layer(out_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout2d(0.1)
        # )

    def _make_stages(self, in_channels, out_channels, bin_sz, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = norm_layer(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)
    
    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages])
        output = torch.cat(pyramids, dim=1)
        return output


class PSP(nn.Module):
	def __init__(self, dim_in, dim_out):
		super(PSP, self).__init__()
		pool_sizes=[1,2,4,8]
		self.branch1 = _PSPModule(dim_in,pool_sizes=pool_sizes,norm_layer=nn.BatchNorm2d)
		self.branch2 = StripPooling(dim_in,out_channels=512,up_kwargs={'mode': 'bilinear', 'align_corners': True})
		self.bottleneck = nn.Sequential(
            nn.Conv2d(dim_in*2+512, dim_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
		#self.psa = ParallelPolarizedSelfAttention(channel=dim_out)
	def forward(self, x):
        #-----------------------------------------#
        #   第1个分支，psp
        #-----------------------------------------#
		psp_feature = self.branch1(x)
        #-----------------------------------------#
        #   第2个分支，条形池化
        #-----------------------------------------#
		SP_feature = self.branch2(x)

		feature_cat = torch.cat([psp_feature,SP_feature], dim=1)
		result = self.bottleneck(feature_cat)
		return result    


class up_concat(nn.Module):
    def __init__(self, in_size, out_size):
        super(up_concat, self).__init__()
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.relu   = nn.ReLU(inplace = True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.relu(outputs)
        return outputs


#反卷积操作
class UpsampleModel(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(UpsampleModel, self).__init__()
        self.conv1 = nn.ConvTranspose2d(
            in_channels=in_channel,  # 输入通道数
            out_channels=out_channel,  # 输出通道数
            kernel_size=3,  # 卷积核大小
            stride=2,  # 步长为2
            padding=1,  # 填充
            output_padding=1  # 输出填充
        )

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        return x

class Unet(nn.Module):
    def __init__(self, num_classes = 3, pretrained = False, backbone = 'vgg'):
        super(Unet, self).__init__()
        if backbone == 'vgg':
            self.vgg    = VGG16(pretrained = pretrained)
            in_filters  = [192, 384, 768, 1024]
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained = pretrained)
            in_filters  = [192, 512, 1024, 3072]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]


        #反卷积操作
        self.upsamle_5 = UpsampleModel(in_channel=2048,out_channel=2048)
        self.upsamle_4 = UpsampleModel(in_channel=512,out_channel=512)
        self.upsamle_3 = UpsampleModel(in_channel=256, out_channel=256)
        self.upsamle_2 = UpsampleModel(in_channel=128, out_channel=128)

        #拼接操作
        self.concat_4 = up_concat(in_size=3072,out_size=512)
        self.concat_3 = up_concat(in_size=1024, out_size=256)
        self.concat_2 = up_concat(in_size=512, out_size=128)
        self.concat_1 = up_concat(in_size=192, out_size=64)

        self.aspp_0 = mini_ASPP(64, dim_out=64)
        self.aspp_1 = mini_ASPP(256, dim_out=256)
        self.aspp_2 = mini_ASPP(512, dim_out=512)
        
        self.psp = PSP(dim_in=1024, dim_out=512)

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor = 2), 
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Sequential(
            nn.Conv2d(out_filters[0], num_classes, 1),
            nn.ReLU(),
        )
        self.aux = nn.Sequential(
            nn.Conv2d(out_filters[0], 2, 1),
            nn.Sigmoid(),
        )

        self.backbone = backbone

    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        """
        feat1--64
        feat2--256
        feat3--512
        feat4--1024
        feat5--2048
        """
        #做个层的特征提取
        feat1 = self.aspp_0(feat1)
        # feat2 = self.aspp_1(feat2)
        # feat3 = self.aspp_2(feat3)

        # #第五层反卷积与第四层拼接
        # feat5 = self.upsamle_5(feat5)
        # feat4 = torch.cat([feat4,feat5], 1)
        # feat4 = self.concat_4(feat4) #512
        
        feat4 = self.psp(feat4)

        #第四层反卷积与第三层拼接
        feat4 = self.upsamle_4(feat4)
        feat3 = torch.cat([feat3,feat4], 1)
        feat3 = self.concat_3(feat3) #256,32,32

        #第三层反卷积与第二层拼接
        feat3 = self.upsamle_3(feat3)
        feat2 = torch.cat([feat2, feat3], 1)
        feat2 = self.concat_2(feat2) #128

        #第二层反卷积与第一层拼接
        feat2 = self.upsamle_2(feat2)
        feat1 = torch.cat([feat1, feat2], 1)
        up1 = self.concat_1(feat1)

        # up4 = self.up_concat4(feat4, feat5)
        # up3 = self.up_concat3(feat3, up4)
        # up2 = self.up_concat2(feat2, up3)
        # up1 = self.up_concat1(feat1, up2)  #64,320,240

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)
        aux = self.aux(up1)
        
        return final,aux

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True
