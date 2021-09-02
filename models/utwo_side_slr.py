# *****************************************************************************
#  Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from .model_utils import MyResample2D


class REBNCONV(nn.Module):
    def __init__(self,in_ch=3,out_ch=3,dirate=1):
        super(REBNCONV,self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self,x):

        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout

## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src,tar):

    src = F.interpolate(src, scale_factor=2, mode='bilinear', align_corners=False)

    return src


### RSU-7 ###
class RSU7(nn.Module):#UNet07DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool5 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d =  self.rebnconv6d(torch.cat((hx7,hx6),1))
        hx6dup = _upsample_like(hx6d,hx5)

        hx5d =  self.rebnconv5d(torch.cat((hx6dup,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-6 ###
class RSU6(nn.Module):#UNet06DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)


        hx5d =  self.rebnconv5d(torch.cat((hx6,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-5 ###
class RSU5(nn.Module):#UNet05DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-4 ###
class RSU4(nn.Module):#UNet04DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-4F ###
class RSU4F(nn.Module):#UNet04FRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx2d = self.rebnconv2d(torch.cat((hx3d,hx2),1))
        hx1d = self.rebnconv1d(torch.cat((hx2d,hx1),1))

        return hx1d + hxin


class utwo_side_slr(nn.Module):
    def __init__(self, args, mean_pix=[109.93, 109.167, 101.455], in_channel=6):
        super(utwo_side_slr, self).__init__()
        self.is_output_flow = False

        # --------------------- encoder --------------------
        # conv1
        self.flow_pred_encoder_layer1 = self.make_flow_pred_encoder_layer(in_channel, 32, 7, 3)
        self.flow_pred_encoder_layer2 = self.make_flow_pred_encoder_layer(32, 64, 5, 2)
        self.flow_pred_encoder_layer3 = self.make_flow_pred_encoder_layer(64, 128)
        self.flow_pred_encoder_layer4 = self.make_flow_pred_encoder_layer(128, 256)
        self.flow_pred_encoder_layer5 = self.make_flow_pred_encoder_layer(256, 512)

        self.flow_pred_bottleneck = self.make_flow_pred_encoder_layer(512, 512)

        self.flow_pred_decoder_layer5 = self.make_flow_pred_decoder_layer(512, 512)
        self.flow_pred_decoder_layer4 = self.make_flow_pred_decoder_layer(1024, 256)
        self.flow_pred_decoder_layer3 = self.make_flow_pred_decoder_layer(512, 128)
        self.flow_pred_decoder_layer2 = self.make_flow_pred_decoder_layer(256, 64)
        self.flow_pred_decoder_layer1 = self.make_flow_pred_decoder_layer(128, 32)

        self.flow_pred_refine_layer = nn.Sequential(
            nn.Conv2d(18, 8, 3, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.1))

        self.forward_flow_conv = nn.Conv2d(8, 2, 1)
        self.backward_flow_conv = nn.Conv2d(8, 2, 1)

        self.flow_interp_refine_layer = nn.Sequential(
            nn.Conv2d(18, 8, 3, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.1))

        self.flow_interp_forward_out_layer = nn.Conv2d(8, 2, 1)
        self.flow_interp_backward_out_layer = nn.Conv2d(8, 2, 1)

        # visibility
        self.flow_interp_vis_layer = nn.Conv2d(8, 1, 1)

        self.resample2d_train = MyResample2D(args.crop_size[1], args.crop_size[0])

        mean_pix = torch.from_numpy(np.array(mean_pix)).float()
        mean_pix = mean_pix.view(1, 3, 1, 1)
        self.register_buffer('mean_pix', mean_pix)

        self.args = args
        self.scale = args.flow_scale

        self.L1_loss = nn.L1Loss()
        self.L2_loss = nn.MSELoss()
        self.ignore_keys = ['vgg', 'grid_w', 'grid_h', 'tlinespace', 'resample2d_train', 'resample2d']
        self.register_buffer('tlinespace', torch.linspace(0, 1, 2 + args.num_interp).float())

        vgg16 = torchvision.models.vgg16(pretrained=True)
        self.vgg16_features = nn.Sequential(*list(vgg16.children())[0][:22])
        for param in self.vgg16_features.parameters():
            param.requires_grad = False

        # loss weights
        self.pix_alpha = 0.8
        self.warp_alpha = 0.4 
        self.vgg16_alpha = 0.005
        self.smooth_alpha = 1.

        self.stage1 = RSU7(16,32,64)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2 = RSU6(192,32,128)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage3 = RSU5(384,64,256)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = RSU4(768,128,512)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5 = RSU4F(1536,256,512)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage6 = RSU4F(1024,256,512)

        # decoder
        self.stage5d = RSU4F(1024,256,512)
        self.stage4d = RSU4(1024,128,256)
        self.stage3d = RSU5(512,64,128)
        self.stage2d = RSU6(256,32,64)
        self.stage1d = RSU7(128,16,64)

        out_ch=3

        self.side1 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side2 = nn.Conv2d(128,out_ch,3,padding=1)
        self.side3 = nn.Conv2d(256,out_ch,3,padding=1)
        self.side4 = nn.Conv2d(512,out_ch,3,padding=1)
        self.side5 = nn.Conv2d(1024,out_ch,3,padding=1)
        self.side6 = nn.Conv2d(512,out_ch,3,padding=1)

        self.side1_r = nn.Conv2d(64,out_ch,3,padding=1)
        self.side2_r = nn.Conv2d(64,out_ch,3,padding=1)
        self.side3_r = nn.Conv2d(128,out_ch,3,padding=1)
        self.side4_r = nn.Conv2d(256,out_ch,3,padding=1)
        self.side5_r = nn.Conv2d(512,out_ch,3,padding=1)
        self.side6_r = nn.Conv2d(512,out_ch,3,padding=1)

        #self.outconv = nn.Conv2d(6*out_ch,out_ch,1)



    def make_flow_pred_encoder_layer(self, in_chn, out_chn, kernel_size=3, padding=1):
        layer = nn.Sequential(
            nn.Conv2d(in_chn, out_chn, kernel_size, padding=padding),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(out_chn, out_chn, kernel_size, padding=padding),
            nn.LeakyReLU(inplace=True, negative_slope=0.1))
        return layer

    def make_flow_pred_decoder_layer(self, in_chn, out_chn):
        layer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_chn, out_chn, 3, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(out_chn, out_chn, 3, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.1))
        return layer

    def make_flow_interp_encoder_layer(self, in_chn, out_chn, kernel_size=3, padding=1):
        layer = nn.Sequential(
            nn.Conv2d(in_chn, out_chn, kernel_size, padding=padding),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(out_chn, out_chn, kernel_size, padding=padding),
            nn.LeakyReLU(inplace=True, negative_slope=0.1))
        return layer

    def make_flow_interp_decoder_layer(self, in_chn, out_chn):
        layer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_chn, out_chn, 3, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(out_chn, out_chn, 3, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.1))
        return layer

    def make_flow_prediction(self, x):

        encoder_out1 = self.flow_pred_encoder_layer1(x)
        encoder_out1_pool = F.avg_pool2d(encoder_out1, 2, stride=2)

        encoder_out2 = self.flow_pred_encoder_layer2(encoder_out1_pool)
        encoder_out2_pool = F.avg_pool2d(encoder_out2, 2, stride=2)

        encoder_out3 = self.flow_pred_encoder_layer3(encoder_out2_pool)
        encoder_out3_pool = F.avg_pool2d(encoder_out3, 2, stride=2)

        encoder_out4 = self.flow_pred_encoder_layer4(encoder_out3_pool)
        encoder_out4_pool = F.avg_pool2d(encoder_out4, 2, stride=2)

        encoder_out5 = self.flow_pred_encoder_layer5(encoder_out4_pool)
        encoder_out5_pool = F.avg_pool2d(encoder_out5, 2, stride=2)

        bottleneck_out = self.flow_pred_bottleneck(encoder_out5_pool)

        decoder_out5 = self.flow_pred_decoder_layer5(bottleneck_out)
        decoder_out5 = torch.cat((encoder_out5, decoder_out5), dim=1)

        decoder_out4 = self.flow_pred_decoder_layer4(decoder_out5)
        decoder_out4 = torch.cat((encoder_out4, decoder_out4), dim=1)

        decoder_out3 = self.flow_pred_decoder_layer3(decoder_out4)
        decoder_out3 = torch.cat((encoder_out3, decoder_out3), dim=1)

        decoder_out2 = self.flow_pred_decoder_layer2(decoder_out3)
        decoder_out2 = torch.cat((encoder_out2, decoder_out2), dim=1)

        decoder_out1 = self.flow_pred_decoder_layer1(decoder_out2)
        decoder_out1 = torch.cat((encoder_out1, decoder_out1), dim=1)

        #side output
        d1 = self.side1(decoder_out1)

        d2 = self.side2(decoder_out2)
        #d2 = _upsample_like(d2,d1)
        d2 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        
        d3 = self.side3(decoder_out3)
        #d3 = _upsample_like(d3,d1)
        d3 = F.interpolate(d3, scale_factor=4, mode='bilinear', align_corners=False)

        d4 = self.side4(decoder_out4)
        #d4 = _upsample_like(d4,d1)
        d4 = F.interpolate(d4, scale_factor=8, mode='bilinear', align_corners=False)

        d5 = self.side5(decoder_out5)
        #d5 = _upsample_like(d5,d1)
        d5 = F.interpolate(d5, scale_factor=16, mode='bilinear', align_corners=False)

        d6 = self.side6(bottleneck_out)
        #d6 = _upsample_like(d6,d1)
        d6 = F.interpolate(d6, scale_factor=32, mode='bilinear', align_corners=False)



        #motion_rep = self.flow_pred_refine_layer(decoder_out1)

        motion_rep = self.flow_pred_refine_layer(torch.cat((d1,d2,d3,d4,d5,d6),1))

        #motion_rep = self.flow_pred_refine_layer(decoder_out1)

        uvf = self.forward_flow_conv(motion_rep)
        uvb = self.backward_flow_conv(motion_rep)

        return uvf, bottleneck_out, uvb, decoder_out5, decoder_out4, decoder_out3, decoder_out2, decoder_out1


    def netuu(self, in_data, flow_pred_bottleneck_out, u_de_5_out, u_de_4_out, u_de_3_out, u_de_2_out, u_de_1_out):

        hx = in_data

        #stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        #stage 2
        #hx2 = self.stage2(hx)
        hx2 = self.stage2(torch.cat((u_de_2_out, hx),1))
        hx = self.pool23(hx2)

        #stage 3
        #hx3 = self.stage3(hx)
        hx3 = self.stage3(torch.cat((u_de_3_out, hx),1))
        hx = self.pool34(hx3)

        #stage 4
        #hx4 = self.stage4(hx)
        hx4 = self.stage4(torch.cat((u_de_4_out, hx),1))
        hx = self.pool45(hx4)

        #stage 5
        #hx5 = self.stage5(hx)
        hx5 = self.stage5(torch.cat((u_de_5_out, hx),1))
        hx = self.pool56(hx5)

        #print('hx:', hx.size())
        #print('flow_pred_bottleneck_out:', flow_pred_bottleneck_out.size())
        #stage 6
        #hx6 = self.stage6(hx)
        hx6 = self.stage6(torch.cat((hx, flow_pred_bottleneck_out), dim=1))

        #hx6 = torch.cat((hx6, flow_pred_bottleneck_out), dim=1)
        hx6up = _upsample_like(hx6,hx5)

        #print('hx6up:', hx6up.size())
        #print('hx5:', hx5.size())
        #-------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.stage4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.stage3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.stage2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))


        d1 = self.side1_r(hx1d)

        d2 = self.side2_r(hx2d)
        #d2 = _upsample_like(d2,d1)
        d2 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        
        d3 = self.side3_r(hx3d)
        #d3 = _upsample_like(d3,d1)
        d3 = F.interpolate(d3, scale_factor=4, mode='bilinear', align_corners=False)

        d4 = self.side4_r(hx4d)
        #d4 = _upsample_like(d4,d1)
        d4 = F.interpolate(d4, scale_factor=8, mode='bilinear', align_corners=False)

        d5 = self.side5_r(hx5d)
        #d5 = _upsample_like(d5,d1)
        d5 = F.interpolate(d5, scale_factor=16, mode='bilinear', align_corners=False)


        d6 = self.side6_r(hx6)
        #d6 = _upsample_like(d6,d1)
        d6 = F.interpolate(d6, scale_factor=32, mode='bilinear', align_corners=False)


        flow_interp_motion_rep = self.flow_interp_refine_layer(torch.cat((d1,d2,d3,d4,d5,d6),1))



        #flow_interp_motion_rep = self.flow_interp_refine_layer(hx1d)

        flow_interp_forward_flow = self.flow_interp_forward_out_layer(flow_interp_motion_rep)
        flow_interp_backward_flow = self.flow_interp_backward_out_layer(flow_interp_motion_rep)

        flow_interp_vis_map = self.flow_interp_vis_layer(flow_interp_motion_rep)
        flow_interp_vis_map = torch.sigmoid(flow_interp_vis_map)

        return flow_interp_forward_flow, flow_interp_backward_flow, flow_interp_vis_map


    def forward(self, inputs, target_index):
        if 'image' in inputs:
            inputs = inputs['image']

        if self.training:
            self.resample2d = self.resample2d_train
        else:
            _, _, height, width = inputs[0].shape
            self.resample2d = MyResample2D(width, height).cuda()
            
        # Normalize inputs
        im1, im_target, im2 = [(im - self.mean_pix) for im in inputs]

        # Estimate bi-directional optical flows between input low FPS frame pairs
        # Downsample images for robust intermediate flow estimation
        ds_im1 = F.interpolate(im1, scale_factor=1./self.scale, mode='bilinear', align_corners=False)
        ds_im2 = F.interpolate(im2, scale_factor=1./self.scale, mode='bilinear', align_corners=False)

        uvf, bottleneck_out, uvb, u_de_5, u_de_4, u_de_3, u_de_2, u_de_1 = self.make_flow_prediction(torch.cat((ds_im1, ds_im2), dim=1))

        uvf = self.scale * F.interpolate(uvf, scale_factor=self.scale, mode='bilinear', align_corners=False)
        uvb = self.scale * F.interpolate(uvb, scale_factor=self.scale, mode='bilinear', align_corners=False)
        bottleneck_out = F.interpolate(bottleneck_out, scale_factor=self.scale, mode='bilinear', align_corners=False)

        t = self.tlinespace[target_index]
        t = t.reshape(t.shape[0], 1, 1, 1)

        uvb_t_raw = - (1 - t) * t * uvf + t * t * uvb
        uvf_t_raw = (1 - t) * (1 - t) * uvf - (1 - t) * t * uvb

        im1w_raw = self.resample2d(im1, uvb_t_raw)  # im1w_raw
        im2w_raw = self.resample2d(im2, uvf_t_raw)  # im2w_raw

        # Perform intermediate bi-directional flow refinement
        uv_t_data = torch.cat((im1, im2, im1w_raw, uvb_t_raw, im2w_raw, uvf_t_raw), dim=1)
        #uvf_t, uvb_t, t_vis_map = self.make_flow_interpolation(uv_t_data, bottleneck_out)

        uvf_t, uvb_t, t_vis_map = self.netuu(uv_t_data, bottleneck_out, u_de_5, u_de_4, u_de_3, u_de_2, u_de_1)


        uvb_t = uvb_t_raw + uvb_t # uvb_t
        uvf_t = uvf_t_raw + uvf_t # uvf_t

        im1w = self.resample2d(im1, uvb_t)  # im1w
        im2w = self.resample2d(im2, uvf_t)  # im2w

        # Compute final intermediate frame via weighted blending
        alpha1 = (1 - t) * t_vis_map
        alpha2 = t * (1 - t_vis_map)
        denorm = alpha1 + alpha2 + 1e-10
        im_t_out = (alpha1 * im1w + alpha2 * im2w) / denorm

        # Calculate training loss
        losses = {}
        losses['pix_loss'] = self.L1_loss(im_t_out, im_target)

        im_t_out_features = self.vgg16_features(im_t_out/255.)
        im_target_features = self.vgg16_features(im_target/255.)
        losses['vgg16_loss'] = self.L2_loss(im_t_out_features, im_target_features)

        losses['warp_loss'] = self.L1_loss(im1w_raw, im_target) + self.L1_loss(im2w_raw, im_target) + \
            self.L1_loss(self.resample2d(im1, uvb.contiguous()), im2) + \
            self.L1_loss(self.resample2d(im2, uvf.contiguous()), im1)

        smooth_bwd = self.L1_loss(uvb[:, :, :, :-1], uvb[:, :, :, 1:]) + \
            self.L1_loss(uvb[:, :, :-1, :], uvb[:, :, 1:, :])
        smooth_fwd = self.L1_loss(uvf[:, :, :, :-1], uvf[:, :, :, 1:]) + \
            self.L1_loss(uvf[:, :, :-1, :], uvf[:, :, 1:, :])

        losses['smooth_loss'] = smooth_bwd + smooth_fwd

        # Coefficients for total loss determined empirically using a validation set
        losses['tot'] = self.pix_alpha * losses['pix_loss'] + self.warp_alpha * losses['warp_loss'] \
            + self.vgg16_alpha * losses['vgg16_loss'] + self.smooth_alpha * losses['smooth_loss']

        # Converts back to (0, 255) range
        im_t_out = im_t_out + self.mean_pix
        im_target = im_target + self.mean_pix

        return losses, im_t_out, im_target
