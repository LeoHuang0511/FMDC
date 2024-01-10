import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.VGG.conv import BasicConv
from .dcn import DeformableConv2d
from .VGG.conv import BasicConv, ResBlock
import pytorch_ssim


class optical_deformable_alignment_module(nn.Module):
    def __init__(self):
        super(optical_deformable_alignment_module, self).__init__()
        self.offset_groups = 4
        self.deformable_kernel_size = 3
        self.padding = (self.deformable_kernel_size - 1) // 2
        self.deformable_conv1 = DeformableConv2d(256, 256, self.offset_groups, kernel_size=self.deformable_kernel_size, padding = self.padding)
        self.deformable_conv2 = DeformableConv2d(256, 256, self.offset_groups, kernel_size=self.deformable_kernel_size, padding = self.padding)
        self.weight_conv =  ResBlock(512, 512) 
        self.reduce_channel2 = ResBlock(512, 256)
        self.reduce_channel = ResBlock(256, 128)
#         self.ssim = pytorch_ssim.SSIM()
        self.offset_loss = 0.
#         self.similarity = nn.CosineSimilarity(dim=1, eps=1e-7)
        
    def forward(self,reference, source): #b c h w
#         pre_warp_ref = optical_flow_warping(reference, -flow)
        batch = source.size(dim = 0)
        ref_refined_feature,offset2sou = self.deformable_conv1(reference, source) #256 
        sour_refined_feature,offset2ref = self.deformable_conv1(source, reference)
        offset2s = F.interpolate(offset2sou,scale_factor=4,mode='bilinear',align_corners=True) * 4 
        offset2r = F.interpolate(offset2ref,scale_factor=4,mode='bilinear',align_corners=True) * 4 
        self.offset_loss = self.deformable_conv1.offset_loss
#         img1 = img[0::2,:,:,:]
#         img2 = img[1::2,:,:,:]
#         print("opti:",ref_refined_feature.shape)
#         print("opti:",sour_refined_feature.shape)
#         self.img1_warp = optical_flow_warping(img1, offset2s)
#         self.img2_warp = optical_flow_warping(img2, offset2r)  #offset2r is forward_flow

#         self.ssim_loss = 2 - self.ssim(img1,self.img2_warp) - self.ssim(img2,self.img1_warp) + F.l1_loss(img1,self.img2_warp) +F.l1_loss(img2,self.img1_warp)

        refcorsou = torch.concat([sour_refined_feature, reference], axis = 1) #512  to compute outflow
        soucorref = torch.concat([ref_refined_feature, source], axis = 1)    # compute inflow
        
        compare = torch.concat([refcorsou, soucorref], axis = 0)  #first half compute out, second half compute in
        comp = self.weight_conv(compare)
        com = self.reduce_channel2(comp)
        compare_result = self.reduce_channel(com)

        return compare_result,offset2r, offset2s
#     def forward(self,reference, flow, source, img):
# #         pre_warp_ref = optical_flow_warping(reference, -flow)
#         pre_refined_feature,offset = self.deformable_conv1(reference, source)
#         full_offset = F.interpolate(offset,scale_factor=4,mode='bilinear',align_corners=True) * 4 
#         img1 = img[0::2,:,:,:]
#         img2 = img[1::2,:,:,:]
#         img1_warp = optical_flow_warping(img2, full_offset)
#         self.ssim_loss = 1 - self.ssim(img1,img1_warp)

#         next_refined_feature,_ = self.deformable_conv2(source, pre_refined_feature)
#         weight_in = torch.concat([pre_refined_feature, next_refined_feature], axis = 1 )
#         weight_in = self.weight_conv(weight_in)
#         weight = self.reduce_channel2(weight_in)
#         weight = torch.sigmoid(weight)

#         weighted_feature = pre_refined_feature * weight + next_refined_feature * (1 - weight)
#         weighted_feature = self.reduce_channel(weighted_feature)
#         weighted_feature = self.reduce_channel(weight)
#         return weighted_feature
    def color(self,reference, flow, source):
        pre_warp_ref = optical_flow_warping(reference, flow)
        pre_refined_feature = self.deformable_conv1(pre_warp_ref, source)
        next_refined_feature = self.deformable_conv2(source, pre_refined_feature)
        weight_in = torch.concat([pre_refined_feature, next_refined_feature], axis = 1 )
        weight_in = self.weight_conv(weight_in)
        
        return weight_in

# +
# class optical_deformable_alignment_module(nn.Module):
#     def __init__(self):
#         super(optical_deformable_alignment_module, self).__init__()
#         self.offset_groups = 4
#         self.deformable_kernel_size = 3
#         self.padding = (self.deformable_kernel_size - 1) // 2
#         self.reduce_channel = BasicConv(256, 128, 1, 1, 0, norm = 'bn', relu =True)
#         self.deformable_conv = DeformableConv2d(128, 128, self.offset_groups, kernel_size=self.deformable_kernel_size, padding = self.padding)
#         self.weight_conv =  ResBlock(256, 128) 
            

#     def forward(self,reference, flow, source):
#         reference = self.reduce_channel(reference)
#         source = self.reduce_channel(source)
#         warp_ref = optical_flow_warping(reference, flow)
#         refined_feature = self.deformable_conv(warp_ref, source)
        
#         weight_in = torch.concat([warp_ref, refined_feature], axis = 1 )
#         weight = self.weight_conv(weight_in)
#         weight = torch.sigmoid(weight)

#         weighted_feature = refined_feature * weight + source * (1 - weight)
        
#         return weighted_feature
# -

def batch_similarity_matrix(vec1,vec2,eps = 1e-7):
    sim_matrix = torch.einsum('bdn,bdm->bnm', vec1, vec2)  #inner product (n,m)
#                     #mdesc0(n,256) mdesc1(m,256) frame1 n peoples frames 2 m peoples
#                     #contrasitve loss更改
    m0 = torch.norm(vec1,dim = 1) #l2norm
    m1 = torch.norm(vec2,dim = 1)
    norm = torch.einsum('bn,bm->bnm',m0,m1) + eps # (n,m)
    exp_term = torch.exp(sim_matrix / (256 ** .5 )/norm)[0]

    return exp_term
def optical_flow_warping(x, flo, pad_mode="border"):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow   zero channel for horizontal first channel for vertical
    pad_mode (optional): ref to https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        "zeros": use 0 for out-of-bound grid locations,
        "border": use border values for out-of-bound grid locations
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1).cuda()
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W).cuda()
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float().cuda()

    vgrid = grid + flo  # warp后，新图每个像素对应原图的位置

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid, padding_mode=pad_mode)

#     mask = torch.ones(x.size()).cuda()
#     mask = F.grid_sample(mask, vgrid)

#     mask[mask < 0.9999] = 0
#     mask[mask > 0] = 1

    return output #* mask
