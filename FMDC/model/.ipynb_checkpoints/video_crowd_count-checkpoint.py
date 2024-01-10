import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .VGG.VGG16_FPN import VGG16_FPN
from .optical_deformable_module import optical_deformable_alignment_module
from .unimatch import UniMatch
from .VGG.conv import BasicDeconv, ResBlock
from model.PreciseRoIPooling.pytorch.prroi_pool.functional import prroi_pool2d
from misc.layer import Gaussianlayer
import cv2
from model.points_from_den import *
BN_MOMENTUM = 0.01

# +
class video_crowd_count(nn.Module):
    def __init__(self,cfg, cfg_data):
        super(video_crowd_count, self).__init__()
        self.Extractor = VGG16_FPN().cuda()
        self.optical_defromable_layer = optical_deformable_alignment_module().cuda()
        self.flownet = UniMatch(feature_channels=128,
                     num_scales = 2,
                     upsample_factor = 4,
                     num_head = 1,
                     ffn_dim_expansion=4,
                     num_transformer_layers = 6,
                     reg_refine= True,
                     task='flow').cuda()
        self.flownet.eval()
        self.Gaussian = Gaussianlayer().cuda()
        
        self.flownet.load_state_dict(torch.load('./unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth')['model'], strict=False)
        
        self.mask_predict_layer = nn.Sequential(
            nn.Dropout2d(0.2),
            ResBlock(in_dim=128, out_dim=64, dilation=0, norm="bn"),
            ResBlock(in_dim=64, out_dim=32, dilation=0, norm="bn"),

            nn.ConvTranspose2d(32, 16, 2, stride=2, padding=0, output_padding=0, bias=False),
            nn.BatchNorm2d(16, momentum=BN_MOMENTUM),
#             nn.ReLU(inplace=True),

            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8, momentum=BN_MOMENTUM),
#             nn.ReLU(inplace=True),

            nn.ConvTranspose2d(8, 4, 2, stride=2, padding=0, output_padding=0, bias=False),
            nn.BatchNorm2d(4, momentum=BN_MOMENTUM),
#             nn.ReLU(inplace=True),

            nn.Conv2d(4, 1, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(inplace=True)
            ).cuda()
        
#         for init in range(3):
#             nn.init.constant_(self.mask_predict_layer[1].model[init].conv.weight,0)
#             nn.init.constant_(self.mask_predict_layer[2].model[init].conv.weight,0)
            
#         for init in range(3,11,2):
#             nn.init.constant_(self.mask_predict_layer[init].weight,0)
#         nn.init.constant_(self.mask_predict_layer[5].bias,-1)
#         nn.init.constant_(self.mask_predict_layer[9].bias,-1)
        self.cfg = cfg
        self.dataset_cfg = cfg_data
        self.radius = self.cfg.ROI_RADIUS
        self.device = torch.cuda.current_device()
        self.feature_scale = 1/4.
        self.get_ROI_and_MatchInfo = get_ROI_and_MatchInfo( self.dataset_cfg .TRAIN_SIZE, self.radius, feature_scale=self.feature_scale)

    @property
    def loss(self):
#         return self.counting_mse_loss, self.batch_mask_loss, self.batch_out_loss , self.batch_in_loss, self.batch_consistency
        return self.counting_mse_loss, self.batch_mask_loss,self.batch_out_loss, \
            self.batch_in_loss, self.batch_constrative_loss, self.optical_defromable_layer.offset_loss 

    def colorization(self, img, target, img_rgb,usepredict=False):
        img = img.cuda()
        
        assert  img.size(0)%2 ==0
        img_pair_num = img.size(0)//2  
        img_input = img.clone()
        img_input[1::2,1,:,:] = img_input[1::2,0,:,:]
        img_input[1::2,2,:,:] = img_input[1::2,0,:,:]
        if usepredict and target["scene_name"] == self.pred_scene:
            img_input[0::2,1:,:] = self.predict_img
        elif usepredict:
            self.pred_scene = target["scene_name"]
        feature, _ = self.Extractor(img_input)
        with torch.no_grad():
            flow = self.flownet(F.interpolate(img_rgb[0::2,:,:,:],scale_factor=0.25,mode='bilinear',align_corners=True), 
                             F.interpolate(img_rgb[1::2,:,:,:],scale_factor=0.25,mode='bilinear',align_corners=True),
                             attn_type='swin',
                             attn_splits_list=[2,4],
                             corr_radius_list=[-1,4],
                             prop_radius_list=[-1,1],
                             num_reg_refine=6,
                             task='flow',
                             pred_bidir_flow=False,
                             )
        
        color = self.optical_defromable_layer.color(feature[0::2,:,:,:].cuda(), flow['flow_preds'][-1].cuda(), feature[1::2,:,:,:].cuda())
        # print("refined_fea", refined_feature.size())
        
#         self.predict_img = color * torch.tensor([i for i in range(256)])
        return color
    def forward(self, img,img_rgb, target): #if frame1 size is [B, C, H, W]
        for i in range(len(target)):
            for key,data in target[i].items():
                if torch.is_tensor(data):
                    target[i][key]=data.cuda()
                    
        assert  img.size(0)%2 ==0
        
        img_pair_num = img.size(0)//2  
        feature, den = self.Extractor(img)

        #feature size is[B, C, H/4, W/4] #den size is same as frame size except channel [B, 1, H, W]
#         with torch.no_grad():
            
#             flow = self.flownet(F.interpolate(img_rgb[0::2,:,:,:],scale_factor=0.25,mode='bilinear',align_corners=True), 
#                              F.interpolate(img_rgb[1::2,:,:,:],scale_factor=0.25,mode='bilinear',align_corners=True),
#                              attn_type='swin',
#                              attn_splits_list=[2,4],
#                              corr_radius_list=[-1,4],
#                              prop_radius_list=[-1,1],
#                              num_reg_refine=6,
#                              task='flow',
#                              pred_bidir_flow=False,
#                              )
#             back_flow = self.flownet(F.interpolate(img_rgb[1::2,:,:,:],scale_factor=0.25,mode='bilinear',align_corners=True), 
#                              F.interpolate(img_rgb[0::2,:,:,:],scale_factor=0.25,mode='bilinear',align_corners=True),
#                              attn_type='swin',
#                              attn_splits_list=[2,4],
#                              corr_radius_list=[-1,4],
#                              prop_radius_list=[-1,1],
#                              num_reg_refine=6,
#                              task='flow',
#                              pred_bidir_flow=False,
#                              )
        
#         compare, flow , back_flow = self.optical_defromable_layer(feature[0::2,:,:,:], flow['flow_preds'][-1], feature[1::2,:,:,:])
        compare, flow , back_flow = self.optical_defromable_layer(feature[0::2,:,:,:], feature[1::2,:,:,:])
#         back_refined_feature = self.optical_defromable_layer(feature[1::2,:,:,:], back_flow['flow_preds'][-1], feature[0::2,:,:,:])
        # print("refined_fea", refined_feature.size())
        f_mask = self.mask_predict_layer(compare)
#         back_mask = self.mask_predict_layer(back_refined_feature)

        mask = torch.sigmoid(f_mask)
#         back_mask = torch.sigmoid(back_mask)
#         mask = torch.cat([f_mask,back_mask],dim=1)
#         mask = torch.sigmoid(f_mask)
        #compute count mse loss
        dot_map = torch.zeros_like(den)
        
        for i, data in enumerate(target):
            points = data['points'].long()
            dot_map[i, 0, points[:, 1], points[:, 0]] = 1
        gt_den = self.Gaussian(dot_map)
        gt_mask = torch.zeros(img_pair_num, 2, den.size(2), den.size(3)).cuda()
        assert den.size() == gt_den.size()
        gt_inflow_cnt = torch.zeros(img_pair_num).detach()
        gt_outflow_cnt = torch.zeros(img_pair_num).detach()
        
        self.counting_mse_loss = F.mse_loss(den, gt_den * self.dataset_cfg.DEN_FACTOR)
        den = den/self.dataset_cfg.DEN_FACTOR
        self.batch_constrative_loss = 0
        forward_offset = torch.zeros_like(gt_mask)
        back_offset = torch.zeros_like(gt_mask)
        for pair_idx in range(img_pair_num):
            count_in_pair=[target[pair_idx * 2]['points'].size(0), target[pair_idx * 2+1]['points'].size(0)]
            
            mask_out = torch.zeros(1, 1, den.size(2), den.size(3)).cuda()
            mask_in = torch.zeros(1, 1, den.size(2), den.size(3)).cuda()
            if (np.array(count_in_pair) > 0).all() and (np.array(count_in_pair) < 4000).all():
                match_gt, pois = self.get_ROI_and_MatchInfo(target[pair_idx * 2], target[pair_idx * 2+1],'ab')
                person_move = target[pair_idx * 2 + 1]['points'][match_gt['a2b'][:,1]]  \
                - target[pair_idx * 2]['points'][match_gt['a2b'][:,0]]  
                #target person head location match_gt is match person index
                for it, personid in enumerate(match_gt['a2b']):
                    center_point = target[pair_idx * 2]['points'][personid[0]]
                    vertical_p = int(center_point[1])
                    horizontal_p = int(center_point[0])
                    forward_offset[pair_idx,0,vertical_p - 8 : vertical_p + 15, horizontal_p - 8 :horizontal_p + 8] = person_move[it,0]
                    forward_offset[pair_idx,1,vertical_p - 8 : vertical_p + 15, horizontal_p - 8 :horizontal_p + 8] = person_move[it,1]
                    center_point = target[pair_idx * 2+1]['points'][personid[1]]
                    vertical_p = int(center_point[1])
                    horizontal_p = int(center_point[0])
                    back_offset[pair_idx,0,vertical_p - 8 : vertical_p + 15, horizontal_p - 8 :horizontal_p + 8] = -person_move[it,0]
                    back_offset[pair_idx,1,vertical_p - 8 : vertical_p + 15, horizontal_p - 8 :horizontal_p + 8] = -person_move[it,1]
#                 for point in target[pair_idx * 2]['points'][match_gt['a2b'][:,0]]:
#                     forward_offset[] = 
#                 print(person_move)
#                 offset[pair_idx,0,,]
                if len(match_gt['a2b'][:, 0]) > 0:
                    poi_features = prroi_pool2d(feature[pair_idx*2:pair_idx*2+2], pois, 1, 1, self.feature_scale)
                    poi_features=  poi_features.squeeze(2).squeeze(2)[None].transpose(1,2) # [batch, dim, num_features]
                    mdesc0, mdesc1 = torch.split(poi_features, count_in_pair,dim=2)
                    sim_matrix = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)  #inner product (n,m)
#                     #mdesc0(n,256) mdesc1(m,256) frame1 n peoples frames 2 m peoples
#                     #contrasitve loss更改
                    m0 = torch.norm(mdesc0,dim = 1) #l2norm
                    m1 = torch.norm(mdesc1,dim = 1)
                    norm = torch.einsum('bn,bm->bnm',m0,m1) + 1e-7 # (n,m)
                    exp_term = torch.exp(sim_matrix / (256 ** .5 )/ norm)[0]
                    try:
                        topk = torch.topk(exp_term[match_gt['a2b'][:,0]],50,dim = 1).values #(c,b) #取幾個negative 記得加回positive
                    except:
                        topk = exp_term[match_gt['a2b'][:,0]]
                    denominator = torch.sum(topk,dim=1)   #分母 denominator
                    numerator = exp_term[match_gt['a2b'][:, 0], match_gt['a2b'][:, 1]]   #分子 numerator  c個重複  c個loss
                    #modify here!!!
                    #topk_loss = torch.topk(-torch.log(nom / den + 1e-7),k)  
                    self.batch_constrative_loss += torch.sum(-torch.log(numerator / denominator +1e-7))

                out_ind = match_gt['un_a']  #inflow people id
                in_ind = match_gt['un_b']   #outflow people id
                
                if len(out_ind) > 0:
                    gt_outflow_cnt[pair_idx] += len(out_ind)
                    mask_out[0, 0, target[pair_idx * 2]['points'][out_ind, 1].long(), target[pair_idx * 2]['points'][out_ind, 0].long()] = 1
                if len(in_ind) > 0:
                    gt_inflow_cnt[pair_idx] += len(in_ind)
                    mask_in[0, 0, target[pair_idx * 2+1]['points'][in_ind, 1].long(), target[pair_idx * 2+1]['points'][in_ind, 0].long()] = 1  
                
                mask_out = self.Gaussian(mask_out) > 0
                mask_in = self.Gaussian(mask_in) > 0

                gt_mask[pair_idx,0,:,:] = mask_out
                gt_mask[pair_idx,1,:,:] = mask_in
                
        self.offset_mappingloss = F.l1_loss(forward_offset,flow,reduction = "none") * (forward_offset > 0) \
        + F.l1_loss(back_offset, back_flow, reduction="none") * (back_offset > 0)
        self.offset_mappingloss = torch.sum(self.offset_mappingloss)
#         self.batch_mask_loss = torch.tensor(0.)
        self.batch_mask_loss = F.binary_cross_entropy(mask[:img_pair_num], gt_mask[:,0:1,:,:],reduction = "mean") + \
                               F.binary_cross_entropy(mask[img_pair_num:], gt_mask[:,1:2,:,:],reduction = "mean")
#         self.batch_mask_loss = 99 * self.batch_mask_loss * gt_den.reshape(2,2,den.size(2),den.size(3)) + self.batch_mask_loss
#         self.batch_mask_loss = torch.mean(self.batch_mask_loss)


# +
#         self.batch_consistency = F.mse_loss(gt_den[0::2,:,:,:].sum() - pre_outflow_map.sum() + pre_inflow_map.sum(), \
#                                             gt_den[1::2,:,:,:].sum())
#         self.batch_consistency = self.batch_consistency + F.mse_loss(den[0::2,:,:,:].sum() - pre_outflow_map.sum() \
        pre_outflow_map = mask[:img_pair_num,:,:,:]* den[0::2,:,:,:].detach() #* (mask[:,0:1,:,:] >= 0.8)
        pre_inflow_map = mask[img_pair_num:,:,:,:] * den[1::2,:,:,:].detach() #* (mask[:,1:2,:,:] >= 0.8)
        
        gt_outflow_map = gt_mask[:,0:1,:,:] * gt_den[0::2,:,:,:] 
        gt_inflow_map = gt_mask[:,1:2,:,:] * gt_den[1::2,:,:,:] 
        
        self.batch_out_loss = F.mse_loss(pre_outflow_map, gt_outflow_map,reduction = 'sum')
#         self.batch_out_loss = 99 * self.batch_out_loss * (gt_den[0::2]>0) + self.batch_out_loss
#         self.batch_out_loss = torch.sum(self.batch_out_loss)
        self.batch_in_loss = F.mse_loss(pre_inflow_map, gt_inflow_map, reduction='sum')
#         self.batch_in_loss = 99 * self.batch_in_loss * (gt_den[1::2]>0) + self.batch_in_loss
#         self.batch_in_loss = torch.sum(self.batch_in_loss)
        
        
        return  den, gt_den, mask, gt_mask, pre_outflow_map.sum(axis=2).sum(axis=2).detach().cpu() , gt_outflow_cnt,\
                pre_inflow_map.sum(axis=2).sum(axis=2).detach().cpu() , gt_inflow_cnt, flow, back_flow

    def test_or_validate(self, img,img_rgb,target):
        with torch.no_grad():
            img = img.cuda()
            img_rgb = img_rgb.cuda()
            assert  img.size(0)%2 ==0
            img_pair_num = img.size(0)//2
            feature, den = self.Extractor(img)
#             flow = self.flownet(F.interpolate(img_rgb[0::2,:,:,:],scale_factor=0.25,mode='bilinear',align_corners=True), 
#                              F.interpolate(img_rgb[1::2,:,:,:],scale_factor=0.25,mode='bilinear',align_corners=True),
#                              attn_type='swin',
#                              attn_splits_list=[2,4],
#                              corr_radius_list=[-1,4],
#                              prop_radius_list=[-1,1],
#                              num_reg_refine=6,
#                              task='flow',
#                              pred_bidir_flow=False,
#                              )
#             back_flow = self.flownet(F.interpolate(img_rgb[1::2,:,:,:],scale_factor=0.25,mode='bilinear',align_corners=True), 
#                              F.interpolate(img_rgb[0::2,:,:,:],scale_factor=0.25,mode='bilinear',align_corners=True),
#                              attn_type='swin',
#                              attn_splits_list=[2,4],
#                              corr_radius_list=[-1,4],
#                              prop_radius_list=[-1,1],
#                              num_reg_refine=6,
#                              task='flow',
#                              pred_bidir_flow=False,
#                              )
            
        #feature size is[B, C, H/4, W/4] #den size is same as frame size except channel [B, 1, H, W]

            
            compare, flow, back_flow = self.optical_defromable_layer(feature[0::2,:,:,:], feature[1::2,:,:,:])

#             back_refined_feature = self.optical_defromable_layer(feature[1::2,:,:,:], back_flow['flow_preds'][-1], feature[0::2,:,:,:])
            f_mask = self.mask_predict_layer(compare)
            mask = torch.sigmoid(f_mask) 
#             mask = torch.sigmoid(f_mask)
#             back_mask = self.mask_predict_layer(back_refined_feature)
#             back_mask = torch.sigmoid(back_mask) 
            
#             mask = torch.cat([f_mask, back_mask],dim=1)
            den = den.clone()/self.dataset_cfg.DEN_FACTOR
            pre_outflow_map = mask[:img_pair_num,:,:,:] * den[0::2,:,:,:] 
            pre_inflow_map = mask[img_pair_num:,:,:,:] * den[1::2,:,:,:] 
            
            if target != None:
                dot_map = torch.zeros_like(den)
                gt_inflow_cnt = np.zeros(img_pair_num)
                gt_outflow_cnt = np.zeros(img_pair_num)
                for i, data in enumerate(target):
                    points = data['points'].long()
                    dot_map[i, 0, points[:, 1], points[:, 0]] = 1
                gt_den = self.Gaussian(dot_map)
                gt_mask = torch.zeros(img_pair_num, 2, den.size(2), den.size(3)).cuda()
                for i in range(len(target)):
                    for key,data in target[i].items():
                        if torch.is_tensor(data):
                            target[i][key]=data.cuda()
                for pair_idx in range(img_pair_num):
                    count_in_pair=[target[pair_idx * 2]['points'].size(0), target[pair_idx * 2+1]['points'].size(0)]
                    if (np.array(count_in_pair) > 0).all() and (np.array(count_in_pair) < 4000).all():
                        match_gt, pois = self.get_ROI_and_MatchInfo(target[pair_idx * 2], target[pair_idx * 2+1],'ab')

                        out_ind = match_gt['un_a']
                        in_ind = match_gt['un_b']
                        mask_out = torch.zeros(1, 1, den.size(2), den.size(3)).cuda()
                        mask_in = torch.zeros(1, 1, den.size(2), den.size(3)).cuda()
                
                        if len(out_ind) > 0:
                            gt_outflow_cnt[pair_idx] += len(out_ind)
                            mask_out[0, 0, target[pair_idx * 2]['points'][out_ind, 1].long(), target[pair_idx * 2]['points'][out_ind, 0].long()] = 1
                        if len(in_ind) > 0:
                            gt_inflow_cnt[pair_idx] += len(in_ind)
                            mask_in[0, 0, target[pair_idx * 2+1]['points'][in_ind, 1].long(), target[pair_idx * 2+1]['points'][in_ind, 0].long()] = 1  
                
                        mask_out = self.Gaussian(mask_out) > 1e-6
                        mask_in = self.Gaussian(mask_in) > 1e-6

                        gt_mask[pair_idx,0,:,:] = mask_out
                        gt_mask[pair_idx,1,:,:] = mask_in
                gt_outflow_map = gt_mask[:,0:1,:,:] * gt_den[0::2,:,:,:] 
                gt_inflow_map = gt_mask[:,1:2,:,:] * gt_den[1::2,:,:,:]
                
                return  den, gt_den, mask, gt_mask, pre_outflow_map.sum().item(), gt_outflow_cnt,\
                pre_inflow_map.sum().item() , gt_inflow_cnt
            else:
                return den, mask, pre_outflow_map.sum().item(), pre_inflow_map.sum().item()
# -


