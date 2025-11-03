import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.xception import xception
from nets.mobilenetv2 import mobilenetv2
from utils.rn import RN_L
from utils.polarized_selfattention import ParallelPolarizedSelfAttention
from utils.stripepool import StripPooling
class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2, self).__init__()
        from functools import partial
        
        model           = mobilenetv2(pretrained)
        self.features   = model.features[:-1]

        self.total_idx  = len(self.features)
        self.down_idx   = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
        
    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:4](x)
        x = self.features[4:](low_level_features)
        return low_level_features, x 


#-----------------------------------------#
#   ASPP特征提取模块
#   利用不同膨胀率的膨胀卷积进行特征提取
#-----------------------------------------#
class ASPP(nn.Module):
	def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
		super(ASPP, self).__init__()
		self.branch1 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate,bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),
		)
		self.branch2 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=6*rate,padding_mode='replicate', dilation=6*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch3 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=12*rate,padding_mode='replicate', dilation=12*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch4 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=18*rate,padding_mode='replicate', dilation=18*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch6 = StripPooling(dim_in,out_channels=dim_out,up_kwargs={'mode': 'bilinear', 'align_corners': True})
		self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0,bias=True)
		self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
		self.branch5_relu = nn.ReLU(inplace=True)
		self.conv_cat = nn.Sequential(
				nn.Conv2d(dim_out*6, dim_out, 1, 1, padding=0,bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),		
		)
        
	def forward(self, x):
		[b, c, row, col] = x.size()
        #-----------------------------------------#
        #   一共五个分支
        #-----------------------------------------#
		conv1x1 = self.branch1(x)
		conv3x3_1 = self.branch2(x)
		conv3x3_2 = self.branch3(x)
		conv3x3_3 = self.branch4(x)
        #-----------------------------------------#
        #   第五个分支，全局平均池化+卷积
        #-----------------------------------------#
		global_feature = torch.mean(x,2,True)
		global_feature = torch.mean(global_feature,3,True)
		global_feature = self.branch5_conv(global_feature)
		global_feature = self.branch5_bn(global_feature)
		global_feature = self.branch5_relu(global_feature)
		global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
		
        #-----------------------------------------#
        #   条形池化
        #-----------------------------------------#
		SP_feature = self.branch6(x)

        #-----------------------------------------#
        #   将五个分支的内容堆叠起来
        #   然后1x1卷积整合特征。
        #-----------------------------------------#
		feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature,SP_feature], dim=1)
		result = self.conv_cat(feature_cat)
		return result

class mini_ASPP(nn.Module):
	def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
		super(mini_ASPP, self).__init__()
		self.branch1 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=1, padding_mode='replicate',bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch2 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=2*rate, padding_mode='replicate', dilation=2*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch3 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=4*rate, padding_mode='replicate', dilation=4*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)

		self.conv_cat = nn.Sequential(
				nn.Conv2d(dim_out*3, dim_out, 1, 1, padding=0,bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),		
		)

	def forward(self, x):
		[b, c, row, col] = x.size()
        #-----------------------------------------#
        #   一共三个分支
        #-----------------------------------------#
		conv3x3_1 = self.branch1(x)
		conv3x3_2 = self.branch2(x)
		conv3x3_4 = self.branch3(x)
		
        #-----------------------------------------#
        #   将三个分支的内容堆叠起来
        #   然后1x1卷积整合特征。
        #-----------------------------------------#
		feature_cat = torch.cat([conv3x3_1, conv3x3_2, conv3x3_4], dim=1)
		result = self.conv_cat(feature_cat)
		return result
class DeepLab(nn.Module):
    def __init__(self, num_classes, backbone="mobilenet", pretrained=True, downsample_factor=16):
        super(DeepLab, self).__init__()
        if backbone=="xception":
            #----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,256]
            #   主干部分    [30,30,2048]
            #----------------------------------#
            self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 2048
            low_level_channels = 256
        elif backbone=="mobilenet":
            #----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,24]
            #   主干部分    [30,30,320]
            #----------------------------------#
            self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320
            low_level_channels = 24
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))
        
        #-----------------------------------------#
        #   ASPP特征提取模块
        #   利用不同膨胀率的膨胀卷积进行特征提取
        #-----------------------------------------#
        self.daspp = ASPP(dim_in=in_channels, dim_out=256, rate=16//downsample_factor)
        self.aspp_1 = mini_ASPP(128,dim_out=48,rate=16//downsample_factor)
        self.aspp_2 = mini_ASPP(low_level_channels,dim_out=32,rate=16//downsample_factor)
        self.aspp_3 = mini_ASPP(728,dim_out=24,rate=16//downsample_factor)
        #----------------------------------#
        #   极化自注意力
        #----------------------------------#
        self.psa = ParallelPolarizedSelfAttention(channel=256)
        self.psa_1 = ParallelPolarizedSelfAttention(channel=48)
        self.psa_2 = ParallelPolarizedSelfAttention(channel=32)
        self.psa_3 = ParallelPolarizedSelfAttention(channel=24)
        #----------------------------------#
        #   转置卷积上采样
        #----------------------------------#
        # self.upsample1=nn.Sequential(
        #     nn.Conv2d(256,256,3,1,1,bias=False),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.2)
        # )
        # self.upsample2=nn.Sequential(
        #     nn.Conv2d(256,256,3,1,1,bias=False),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.2)
        # )
        self.upsample3=nn.Sequential(
            nn.ConvTranspose2d(num_classes,num_classes,4,2,1,bias=False),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_classes,num_classes,3,1,padding=1,padding_mode='replicate',bias=False),
        )
        # self.shortcut_conv = nn.Sequential(
        #     nn.Conv2d(low_level_channels, 48, 1),
        #     nn.BatchNorm2d(48),
        #     nn.ReLU(inplace=True)
        # )
        
        self.up_conv_1 = nn.Sequential(
             nn.Conv2d(256+24,256,3,1,1,bias=False,padding_mode='replicate'),
             nn.BatchNorm2d(256),
             nn.ReLU(inplace=True),
             nn.Dropout(0.4)
        )
        self.up_conv_2 = nn.Sequential(
             nn.Conv2d(256+32,256,3,1,1,bias=False,padding_mode='replicate'),
             nn.BatchNorm2d(256),
             nn.ReLU(inplace=True),
             nn.Dropout(0.4)
        )
        self.up_conv_3 = nn.Sequential(
            nn.Conv2d(48+256, 256, 3, stride=1, padding=1,bias=False,padding_mode='replicate'),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Conv2d(256, 256, 3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.2),
        )
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        #-----------------------------------------#
        #   获得三个特征层
        #   raw_pp: 输入图像进行mini_ASPP结构提取小尺度特征
        #   low_level_features: 浅层特征-进行卷积处理
        #   x : 主干部分-利用ASPP结构进行加强特征提取
        #-----------------------------------------#
        feature_layer_1,feature_layer_2,feature_layer_3,x=self.backbone(x)
        x = self.daspp(x)
        x = self.psa(x)
        
        feature_layer_1 = self.aspp_1(feature_layer_1)
        feature_layer_1 = self.psa_1(feature_layer_1)
        
        feature_layer_2 = self.aspp_2(feature_layer_2)
        feature_layer_2 = self.psa_2(feature_layer_2)

        feature_layer_3 = self.aspp_3(feature_layer_3)
        feature_layer_3 = self.psa_3(feature_layer_3)
        #-----------------------------------------#
        #   将加强特征边上采样
        #   与浅层特征堆叠后利用卷积进行特征提取
        #-----------------------------------------#
        up_layer_1 = self.up_conv_1(torch.cat((x,feature_layer_3),dim=1))
        up_layer_1 = F.interpolate(up_layer_1, size=(feature_layer_2.size(2), feature_layer_2.size(3)), mode='bilinear', align_corners=True)
        #up_layer_1 = self.upsample1(up_layer_1)
        up_layer_2 = self.up_conv_2(torch.cat((up_layer_1,feature_layer_2),dim=1))
        up_layer_2 = F.interpolate(up_layer_2,size=(feature_layer_1.size(2), feature_layer_1.size(3)),mode='bilinear',align_corners=True)
        #up_layer_2 = self.upsample2(up_layer_2)
        up_layer_3 = self.up_conv_3(torch.cat((up_layer_2,feature_layer_1), dim=1))
        final_layer = self.cls_conv(up_layer_3)
        final_layer = self.upsample3(final_layer)
        return final_layer

