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


class conv_block_nested(nn.Module):
    
    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        return output


class uplus(nn.Module):
    def __init__(self, args, mean_pix=[109.93, 109.167, 101.455], in_channel=6):
        super(uplus, self).__init__()
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
            nn.Conv2d(64, 32, 3, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.1))

        self.forward_flow_conv = nn.Conv2d(32, 2, 1)
        self.backward_flow_conv = nn.Conv2d(32, 2, 1)

        # -------------- flow interpolation encoder-decoder --------------
        #self.flow_interp_encoder_layer1 = self.make_flow_interp_encoder_layer(16, 32, 7, 3)
        #self.flow_interp_encoder_layer2 = self.make_flow_interp_encoder_layer(32, 64, 5, 2)
        #self.flow_interp_encoder_layer3 = self.make_flow_interp_encoder_layer(64, 128)
        #self.flow_interp_encoder_layer4 = self.make_flow_interp_encoder_layer(128, 256)
        #self.flow_interp_encoder_layer5 = self.make_flow_interp_encoder_layer(256, 512)

        #self.flow_interp_bottleneck = self.make_flow_interp_encoder_layer(512, 512)

        #self.flow_interp_decoder_layer5 = self.make_flow_interp_decoder_layer(1024, 512)
        #self.flow_interp_decoder_layer4 = self.make_flow_interp_decoder_layer(1024, 256)
        #self.flow_interp_decoder_layer3 = self.make_flow_interp_decoder_layer(512, 128)
        #self.flow_interp_decoder_layer2 = self.make_flow_interp_decoder_layer(256, 64)
        #self.flow_interp_decoder_layer1 = self.make_flow_interp_decoder_layer(128, 32)

        self.flow_interp_refine_layer = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.1))

        self.flow_interp_forward_out_layer = nn.Conv2d(32, 2, 1)
        self.flow_interp_backward_out_layer = nn.Conv2d(32, 2, 1)

        # visibility
        self.flow_interp_vis_layer = nn.Conv2d(32, 1, 1)

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

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16, n1 * 32]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = conv_block_nested(16, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])
        self.conv5_0 = conv_block_nested(filters[4], filters[5], filters[5])

        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3])
        #self.conv4_1 = conv_block_nested(filters[4] + filters[5], filters[4], filters[4])
        self.conv4_1 = conv_block_nested(2048, filters[4], filters[4])

        self.conv0_2 = conv_block_nested(filters[0]*2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1]*2 + filters[2], filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2]*2 + filters[3], filters[2], filters[2])
        self.conv3_2 = conv_block_nested(filters[3]*2 + filters[4], filters[3], filters[3])

        self.conv0_3 = conv_block_nested(filters[0]*3 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1]*3 + filters[2], filters[1], filters[1])
        self.conv2_3 = conv_block_nested(filters[2]*3 + filters[3], filters[2], filters[2])

        self.conv0_4 = conv_block_nested(filters[0]*4 + filters[1], filters[0], filters[0])
        self.conv1_4 = conv_block_nested(filters[1]*4 + filters[2], filters[1], filters[1])

        self.conv0_5 = conv_block_nested(filters[0]*5 + filters[1], filters[0], filters[0])

        self.final = nn.Conv2d(filters[0], 64, kernel_size=1)



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

        motion_rep = self.flow_pred_refine_layer(decoder_out1)

        uvf = self.forward_flow_conv(motion_rep)
        uvb = self.backward_flow_conv(motion_rep)

        return uvf, bottleneck_out, uvb

	
    def nest2(self, in_data, flow_pred_bottleneck_out):

        x0_0 = self.conv0_0(in_data)

        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))


        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))

        x5_0 = self.conv5_0(self.pool(x4_0))
        flow_interp_bottleneck_out = torch.cat((flow_pred_bottleneck_out, x5_0), dim=1)

        x4_1 = self.conv4_1(torch.cat([x4_0, self.Up(flow_interp_bottleneck_out)], 1))
        x3_2 = self.conv3_2(torch.cat([x3_0, x3_1, self.Up(x4_1)], 1))
        x2_3 = self.conv2_3(torch.cat([x2_0, x2_1, x2_2, self.Up(x3_2)], 1))
        x1_4 = self.conv1_4(torch.cat([x1_0, x1_1, x1_2, x1_3, self.Up(x2_3)], 1))
        x0_5 = self.conv0_5(torch.cat([x0_0, x0_1, x0_2, x0_3, x0_4, self.Up(x1_4)], 1))

        output = self.final(x0_5)

        flow_interp_motion_rep = self.flow_interp_refine_layer(output)

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
        #print('-------------------------------------')
        # Normalize inputs
        im1, im_target, im2 = [(im - self.mean_pix) for im in inputs]

        # Estimate bi-directional optical flows between input low FPS frame pairs
        # Downsample images for robust intermediate flow estimation
        ds_im1 = F.interpolate(im1, scale_factor=1./self.scale, mode='bilinear', align_corners=False)
        ds_im2 = F.interpolate(im2, scale_factor=1./self.scale, mode='bilinear', align_corners=False)

        uvf, bottleneck_out, uvb = self.make_flow_prediction(torch.cat((ds_im1, ds_im2), dim=1))

        #print('bottleneck_out:',bottleneck_out.size())

        uvf = self.scale * F.interpolate(uvf, scale_factor=self.scale, mode='bilinear', align_corners=False)
        uvb = self.scale * F.interpolate(uvb, scale_factor=self.scale, mode='bilinear', align_corners=False)
        bottleneck_out = F.interpolate(bottleneck_out, scale_factor=self.scale, mode='bilinear', align_corners=False)

        #print('bottleneck_out:',bottleneck_out.size())

        t = self.tlinespace[target_index]
        t = t.reshape(t.shape[0], 1, 1, 1)

        uvb_t_raw = - (1 - t) * t * uvf + t * t * uvb
        uvf_t_raw = (1 - t) * (1 - t) * uvf - (1 - t) * t * uvb

        im1w_raw = self.resample2d(im1, uvb_t_raw)  # im1w_raw
        im2w_raw = self.resample2d(im2, uvf_t_raw)  # im2w_raw

        # Perform intermediate bi-directional flow refinement
        uv_t_data = torch.cat((im1, im2, im1w_raw, uvb_t_raw, im2w_raw, uvf_t_raw), dim=1)

        #print('uv_t_data:',uv_t_data.size())

        #uvf_t, uvb_t, t_vis_map = self.make_flow_interpolation(uv_t_data, bottleneck_out)

        uvf_t, uvb_t, t_vis_map = self.nest2(uv_t_data, bottleneck_out)
        #print('')


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
