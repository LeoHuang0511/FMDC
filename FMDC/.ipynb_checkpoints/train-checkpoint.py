# +
import numpy as np
import torch
from torch import optim
import datasets
from misc.utils import *
# from model.VIC import Video_Individual_Counter
from model.video_crowd_count import video_crowd_count
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.cm as cm
from pathlib import Path
from config import cfg
from misc.KPI_pool import Task_KPI_Pool
from thop import profile
class Trainer():
    def __init__(self, cfg_data, pwd):
        self.exp_name = cfg.EXP_NAME
        self.exp_path = cfg.EXP_PATH
        self.pwd = pwd
        self.cfg_data = cfg_data
        self.net = video_crowd_count(cfg, cfg_data)
        
        
        
        self.train_loader, self.color_loader, self.val_loader, self.restore_transform = datasets.loading_data(cfg.DATASET, cfg.VAL_INTERVALS)
#         self.test_loader, _ = datasets.loading_testset(cfg.DATASET, test_interval=cfg.VAL_INTERVALS, mode='test')
        self.use_pred = False
        params = [
            {"params": self.net.Extractor.parameters(), 'lr': cfg.LR_Base, 'weight_decay': cfg.WEIGHT_DECAY},
            {"params": self.net.optical_defromable_layer.parameters(), "lr": cfg.LR_Thre, 'weight_decay': cfg.WEIGHT_DECAY},
            {"params": self.net.mask_predict_layer.parameters(), "lr": cfg.LR_Thre, 'weight_decay': cfg.WEIGHT_DECAY},
        ]
        self.optimizer = optim.Adam(params)
        self.i_tb = 0
        self.epoch = 1
        self.timer={'iter time': Timer(), 'train time': Timer(), 'val time': Timer()}

        self.num_iters = cfg.MAX_EPOCH * np.int64(len(self.train_loader))
        if cfg.task == "LAB":
            self.train_record = {'best_model_name': '', 'color_loss': 1e20, 'mae': 1e20, 'mse': 1e20, 'seq_MAE':1e20, 'WRAE':1e20, 'MIAE': 1e20, 'MOAE': 1e20}
        else:
            self.train_record = {'best_model_name': '', 'mae': 1e20, 'mse': 1e20, 'seq_MAE':1e20, 'WRAE':1e20, 'MIAE': 1e20, 'MOAE': 1e20}

        if cfg.RESUME:
            latest_state = torch.load(cfg.RESUME_PATH)
#             self.net.load_state_dict(latest_state['net'], strict=True)
            self.net.load_state_dict(latest_state, strict=True)
#             self.optimizer.load_state_dict(latest_state['optimizer'])
#             self.epoch = latest_state['epoch']
#             self.i_tb = latest_state['i_tb']
#             self.train_record = latest_state['train_record']
#             self.exp_path = latest_state['exp_path']
#             self.exp_name = latest_state['exp_name']
            print("Finish loading resume mode")
        self.writer, self.log_txt = logger(self.exp_path, self.exp_name, self.pwd, ['exp','eval','figure','img', 'vis','output'], resume=cfg.RESUME)
#         self.task_KPI=Task_KPI_Pool(task_setting={'den': ['gt_cnt', 'mae'], 'match': ['gt_pairs', 'pre_pairs']}, maximum_sample=1000)
        self.task_KPI=Task_KPI_Pool(task_setting={'den': ['gt_cnt', 'pre_cnt'], 'mask': ['gt_cnt', 'acc_cnt']}, maximum_sample=1000)
    def forward(self):
        for epoch in range(self.epoch, cfg.MAX_EPOCH):
#             self.validate()
            self.epoch = epoch
            self.timer['train time'].tic()
            self.train()
            self.timer['train time'].toc(average=False)
            print( 'train time: {:.2f}s'.format(self.timer['train time'].diff) )
            print( '='*20 )

        
    def train(self): # training for all datasets
        self.net.train()
        lr1, lr2 = adjust_learning_rate(self.optimizer,
                                   self.epoch,
                                   cfg.LR_Base,
                                   cfg.LR_Thre,
                                   cfg.LR_DECAY)
        if cfg.task == "LAB":
            batch_loss = {'mask':AverageMeter()}
        else:
            batch_loss = {'in':AverageMeter(), 'den':AverageMeter(), 'out':AverageMeter(), 'mask':AverageMeter()}
        if cfg.continuous:
            loader = self.color_loader
        else:
            loader = self.train_loader
        for i, data in enumerate(loader, 0):
            self.timer['iter time'].tic()
            self.i_tb += 1
            img, img_rgb,label = data
            
            img = torch.stack(img, 0).cuda()
            img_rgb = torch.stack(img_rgb, 0).cuda()
            if cfg.task == "LAB":
                res = []
                for j in range(1,img.size(0),2):
                    r = np.array(self.restore_transform(img[j].detach().clone()))
                    r = torch.from_numpy(r).permute(2,0,1)
                    res.append(r)
                res = torch.stack(res, 0).cuda()
                color = self.net.colorization(img, img_rgb,label)
                
                a = F.interpolate(color[:,:256,:,:],scale_factor=4)
                b = F.interpolate(color[:,256:,:,:],scale_factor=4)
                loss = F.cross_entropy(a, res[:,1,:,:].long()) + F.cross_entropy(b, res[:,2,:,:].long())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                batch_loss['mask'].update(loss.item())
                if (self.i_tb) % cfg.PRINT_FREQ == 0:
                    self.writer.add_scalar('mask_loss', batch_loss['mask'].avg, self.i_tb)
                    self.timer['iter time'].toc(average=False)
                    print('[ep %d][it %d][loss_mask %.4f][%.2fs]' % \
                        (self.epoch, self.i_tb,  batch_loss['mask'].avg ,self.timer['iter time'].diff))
                if (self.i_tb) % 800 == 0:
                    with torch.no_grad():
                        mask = torch.concat([a[0:1].argmax(dim=1),b[0:1].argmax(dim=1)], axis = 0 )
                        mask = mask.permute(1,2,0)
                        
                        save_results_color(self.i_tb, self.exp_path, self.restore_transform, img[0].clone().unsqueeze(0), 
                        img[1].clone().unsqueeze(0), mask.clone().cpu())
            else:    
                
                
                den, gt_den, mask, gt_mask, pre_out_cnt, gt_out_cnt, pre_inf_cnt, gt_in_cnt,f_flow,b_flow = self.net(img,img_rgb,label)
#             counting_mse_loss,mask_loss,out_loss, in_loss, consist_loss  = self.net.loss
                counting_mse_loss,mask_loss,out_loss, in_loss, con_loss, offset_loss = self.net.loss
#                 counting_mse_loss,out_loss, in_loss, con_loss = self.net.loss

                pre_cnt = den.sum()
                gt_cnt = gt_den.sum()
            
                self.task_KPI.add({'den': {'gt_cnt': gt_cnt, 'pre_cnt': max(0,gt_cnt - (pre_cnt - gt_cnt).abs()) },
                               'mask': {'gt_cnt' : gt_out_cnt.sum()+gt_in_cnt.sum(), 'acc_cnt': \
                                        max(0,gt_out_cnt.sum()+gt_in_cnt.sum() - (pre_inf_cnt - gt_in_cnt).abs().sum() \
                                            - (pre_out_cnt - gt_out_cnt).abs().sum()) }})
                self.KPI = self.task_KPI.query()

                loss = torch.stack([counting_mse_loss  , out_loss + in_loss + mask_loss])
#                 loss = torch.stack([counting_mse_loss ,  out_loss+in_loss ])
                weight = torch.stack([self.KPI['den'],self.KPI['mask']]).to(loss.device)
                weight = -(1-weight) * torch.log(weight+1e-8)
                self.weight = weight/weight.sum()

                all_loss = (self.weight*loss +offset_loss*0.1+ con_loss *0.1).sum()

                self.optimizer.zero_grad()
                all_loss.backward()
                self.optimizer.step()
                batch_loss['in'].update(in_loss.item())
                batch_loss['den'].update(counting_mse_loss.item())
                batch_loss['out'].update(out_loss.item())
                batch_loss['mask'].update(mask_loss.item())

                if (self.i_tb) % cfg.PRINT_FREQ == 0:
                    self.writer.add_scalar('count_mse',batch_loss['den'].avg, self.i_tb)
                    self.writer.add_scalar('mask_loss', batch_loss['mask'].avg, self.i_tb)
                    self.writer.add_scalar('loss_in', batch_loss['in'].avg, self.i_tb)
                    self.writer.add_scalar('loss_out', batch_loss['out'].avg, self.i_tb)
                    self.timer['iter time'].toc(average=False)
                    print('[ep %d][it %d][loss_reg %.4f][loss_mask %.4f][loss_in %.4f][loss_out %.4f][%.2fs]' % \
                        (self.epoch, self.i_tb, batch_loss['den'].avg, batch_loss['mask'].avg,batch_loss['in'].avg,
                        batch_loss['out'].avg,self.timer['iter time'].diff))
#                 print('       [cnt: gt: %.1f pred: %.1f max_pre: %.1f max_gt: %.1f]  '
#                       '[match_gt: %.1f matched_a2b: %.1f ]'
#                       '[gt_count_diff: %.1f pre_count_diff: %.1f] '  %
#                       (gt_cnt.item(), pre_cnt.item(),pre_map.max().item()*cfg_data.DEN_FACTOR, gt_map.max().item()*cfg_data.DEN_FACTOR,\
#                         torch.cat(matched_results['gt_matched']).size(0),TP,
#                         matched_results['gt_count_diff'], matched_results['pre_count_diff']  ))
                if (self.i_tb) % 800== 0:
#                     warp1 = self.net.optical_defromable_layer.img1_warp[0:1].detach().clone()
#                     warp0 = self.net.optical_defromable_layer.img2_warp[0:1].detach().clone()
#                 kpts0 = label[0]['points'].cpu().numpy()  # h,w-> w,h
#                 kpts1 = label[1]['points'].cpu().numpy()  # h,w-> w,h
#                 id0 = label[0]['person_id'].cpu().numpy()
#                 id1 = label[1]['person_id'].cpu().numpy()
#                 matches = matched_results['matches0'][0].cpu().detach().numpy()
#                 confidence = matched_results['matching_scores0'][0].cpu().detach().numpy()
#                 if kpts0.shape[0] > 0 and kpts1.shape[0] > 0:
#                     save_visImg(kpts0, kpts1,  matches, confidence, self.i_tb, img[0].clone(), img[1].clone(), 1,
#                                 self.exp_path, id0, id1,scene_id='',restore_transform=self.restore_transform)
#                     save_warp(self.i_tb,self.exp_path,self.restore_transform,warp0,warp1)
                    save_results_mask(self.i_tb, self.exp_path, self.restore_transform, img[0].clone().unsqueeze(0), \
                                  img[1].clone().unsqueeze(0), den[0].detach().cpu().numpy() , \
                                  gt_den[0].detach().cpu().numpy(), den[1].detach().cpu().numpy(), gt_den[1].detach().cpu().numpy() , \
                                  mask[0,:,:,:].detach().cpu().numpy(), gt_mask[0,0:1,:,:].detach().cpu().numpy(), \
                                  mask[img.size(0)//2,:,:,:].detach().cpu().numpy(), gt_mask[0,1:2,:,:].detach().cpu().numpy(),\
                                     f_flow[0].permute(1,2,0).detach().cpu().numpy(),b_flow[0].permute(1,2,0).detach().cpu().numpy())

            if self.i_tb % 1000== 0 :
                self.timer['val time'].tic()
                self.validate()
                self.net.train()
                self.net.flownet.eval()
                self.timer['val time'].toc(average=False)
                print('val time: {:.2f}s'.format(self.timer['val time'].diff))
    def validate(self):
        self.net.eval()
        sing_cnt_errors = {'mae': AverageMeter(), 'mse': AverageMeter(), 'color':AverageMeter()}
        scenes_pred_dict = []
        scenes_gt_dict = []
#         gt_flow_cnt = [133,737,734,1040,321]
        for scene_id, sub_valset in  enumerate(self.val_loader, 0):
#         for scene_id, sub_valset in  enumerate(self.test_loader, 0):
            gen_tqdm = tqdm(sub_valset)
            video_time = len(sub_valset)+cfg.VAL_INTERVALS
            print(video_time)
            pred_dict = {'id': scene_id, 'time':video_time, 'first_frame': 0, 'inflow': [], 'outflow': []}
            gt_dict  = {'id': scene_id, 'time':video_time, 'first_frame': 0, 'inflow': [], 'outflow': []}
            for vi, data in enumerate(gen_tqdm, 0):
                img, img_rgb, target = data
                img, img_rgb, target = img[0], img_rgb[0],target[0]
#                 img, img_rgb = img[0], img_rgb[0]
                
                img = torch.stack(img,0).cuda()
                img_rgb = torch.stack(img_rgb,0).cuda()
                
                with torch.no_grad():
                    b, c, h, w = img.shape
                    if h % 64 != 0: pad_h = 64 - h % 64
                    else: pad_h = 0
                    if w % 64 != 0: pad_w = 64 - w % 64
                    else: pad_w = 0
                    pad_dims = (0, pad_w, 0, pad_h)
                    img = F.pad(img, pad_dims, "constant")
                    img_rgb = F.pad(img_rgb, pad_dims, "constant")
                    if vi % cfg.VAL_INTERVALS== 0 or vi ==len(sub_valset)-1:
                        frame_signal = 'match'
                    else: frame_signal = 'skip'

                    if frame_signal == 'skip':
                        continue
                    elif cfg.task =="LAB":
                        res = np.array(self.restore_transform(img[1].detach().clone()))
                        res = torch.from_numpy(res).permute(2,0,1).unsqueeze(0).cuda()
                        res = torch.stack(res, 0).cuda()
                        color = self.net.colorization(img, img_rgb,target)
                        a = F.interpolate(color[:,:256,:,:],scale_factor=4)
                        b = F.interpolate(color[:,256:,:,:],scale_factor=4)
                        
                        loss = F.cross_entropy(a[:,:,:h,:w], res[:,1,:,:].long()) + F.cross_entropy(b[:,:,:h,:w], res[:,2,:,:].long())
                        sing_cnt_errors['color'].update(loss.item())
                    else:

                        den, gt_den, mask, gt_mask, pre_out_cnt, gt_out_cnt, pre_inf_cnt, gt_in_cnt \
                        = self.net.test_or_validate(img ,img_rgb,target)
#                         den, _, pre_out_cnt, pre_inf_cnt= self.net.test_or_validate(img, img_rgb,None)
#                         pred_cnt = den[0].sum().item()
                        #    -----------Counting performance------------------
                        gt_count, pred_cnt = gt_den[0].sum().item(),  den[0].sum().item()
                        

                        s_mae = abs(gt_count - pred_cnt)
                        s_mse = ((gt_count - pred_cnt) * (gt_count - pred_cnt))
                        sing_cnt_errors['mae'].update(s_mae)
                        sing_cnt_errors['mse'].update(s_mse)

                        if vi == 0:
                            pred_dict['first_frame'] = den[0].sum().item()
                            gt_dict['first_frame'] = len(target[0]['person_id'])

#                         pred_dict['inflow'].append(matched_results['pre_inflow'])
#                         pred_dict['outflow'].append(matched_results['pre_outflow'])
                        

                        pred_dict['inflow'].append(pre_inf_cnt)
                        pred_dict['outflow'].append(pre_out_cnt)
                        gt_dict['inflow'].append(torch.tensor(gt_in_cnt))
                        gt_dict['outflow'].append(torch.tensor(gt_out_cnt))

                        if frame_signal == 'match':
                            pre_crowdflow_cnt, gt_crowdflow_cnt,_,_ =compute_metrics_single_scene(pred_dict, gt_dict,1)# cfg.VAL_INTERVALS)

                            print('den_gt: %.2f den_pre: %.2f mae: %.2f gt_crowd_flow: %.2f, pre_crowd_flow: %.2f gt_inflow: %.2f pre_inflow:%.2f'
                                  %(gt_count,pred_cnt, s_mae,gt_crowdflow_cnt,pre_crowdflow_cnt, gt_in_cnt,pre_inf_cnt))
#                             pre_crowdflow_cnt, _, _ = compute_metrics_single_scene(pred_dict, 1)

#                             print(' den_pre:  %.2f pre_crowd_flow: %.2f pre_inflow:%.2f'  %  (pred_cnt, pre_crowdflow_cnt, pre_inf_cnt))

#                             kpts0 = matched_results['pre_points'][0][:, 2:4].cpu().numpy()
#                             kpts1 = matched_results['pre_points'][1][:, 2:4].cpu().numpy()

#                             matches = matched_results['matches0'].cpu().numpy()
#                             confidence = matched_results['matching_scores0'].cpu().numpy()
                            # if kpts0.shape[0]>0 and kpts1.shape[0]>0:
                            #     save_visImg(kpts0,kpts1,matches, confidence, vi, img[0].clone(),img[1].clone(),
                            #                      cfg.VAL_INTERVALS, cfg.VAL_VIS_PATH,None,None,scene_id,self.restore_transform)
            if cfg.task != "LAB":
                scenes_pred_dict.append(pred_dict)
                scenes_gt_dict.append(gt_dict)
        # import pdb
        # pdb.set_trace()
        if cfg.task != "LAB":
            MAE, MSE,WRAE, MIAE, MOAE, cnt_result =compute_metrics_all_scenes(scenes_pred_dict,scenes_gt_dict, 1)#cfg.VAL_INTERVALS)
            print('MAE: %.2f, MSE: %.2f  WRAE: %.2f WIAE: %.2f WOAE: %.2f' % (MAE.data, MSE.data, WRAE.data, MIAE.data, MOAE.data))
            print('Pre vs GT:', cnt_result)
            mae = sing_cnt_errors['mae'].avg
            mse = np.sqrt(sing_cnt_errors['mse'].avg)

            self.train_record = update_model(self,{'mae':mae, 'mse':mse, 'seq_MAE':MAE, 'WRAE':WRAE, 'MIAE': MIAE, 'MOAE': MOAE })

            print_NWPU_summary_det(self,{'mae':mae, 'mse':mse, 'seq_MAE':MAE, 'WRAE':WRAE, 'MIAE': MIAE, 'MOAE': MOAE})
#         if cfg.task!="LAB":
#             MAE,MSE, WRAE, crowdflow_cnt  = compute_metrics_all_scenes(scenes_pred_dict, gt_flow_cnt, 1)

#             self.train_record = update_model(self,{'mae':10., 'mse':MSE, 'seq_MAE':MAE, 'WRAE':WRAE, 'MIAE': 10., 'MOAE': 10. })
#             print('MAE: %.2f, MSE: %.2f  WRAE: %.2f' % (MAE.data, MSE.data, WRAE.data))
#             print(crowdflow_cnt)
        else:
            print('color_val_loss: %.5f' % (sing_cnt_errors['color'].avg))
            self.writer.add_scalar('val_color_loss', sing_cnt_errors['color'].avg, self.i_tb)
            if sing_cnt_errors['color'].avg < self.train_record['color_loss']:
                self.train_record['color_loss'] = sing_cnt_errors['color'].avg
                snapshot_name = 'ep_%d_iter_%d_loss_%.3f'% (self.epoch,self.i_tb, self.train_record['color_loss'])
                self.train_record['best_model_name'] = snapshot_name
                torch.save(self.net.state_dict(), os.path.join(self.exp_path, self.exp_name, snapshot_name + '.pth'))
            latest_state = {'train_record':self.train_record, 'net':self.net.state_dict(), 'optimizer':self.optimizer.state_dict(),
                    'epoch': self.epoch, 'i_tb':self.i_tb, 'num_iters':self.num_iters,\
                    'exp_path':self.exp_path, 'exp_name':self.exp_name}
            torch.save(latest_state,os.path.join(self.exp_path, self.exp_name, 'latest_state.pth'))
            

def save_visImg(kpts0,kpts1,matches,confidence,vi, last_frame, cur_frame,intervals,
                save_path,  id0=None, id1=None,scene_id='',restore_transform=None):

    valid = matches > -1
    mkpts0 = kpts0[valid].reshape(-1,2)
    mkpts1 = kpts1[matches[valid]].reshape(-1,2)
    color = cm.jet(confidence[valid])

    text = [
        'VCC',
        'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
        'Matches: {}'.format(len(mkpts1))
    ]
    small_text = [
        'Match Threshold: {:.2f}'.format(0.1),
        'Image Pair: {:06}:{:06}'.format(vi - intervals, vi)
    ]

    out, out_by_point = make_matching_plot_fast(
        last_frame, cur_frame, kpts0, kpts1, mkpts0, mkpts1, color, text,
        path=None, show_keypoints=True, small_text=small_text, restore_transform=restore_transform,
        id0=id0, id1=id1)
    if save_path is not None:

        Path(save_path).mkdir(exist_ok=True)

        stem = '{}_{}_{}_matches'.format(scene_id, vi , vi+ intervals)
        out_file = str(Path(save_path, stem + '.png'))
        print('\nWriting image to {}'.format(out_file))
        cv2.imwrite(out_file, out)
        out_file = str(Path(save_path, stem + '_vis.png'))
        cv2.imwrite(out_file, out_by_point)

def compute_metrics_single_scene(pre_dict, gt_dict, intervals):
    pair_cnt = len(pre_dict['inflow'])
    inflow_cnt, outflow_cnt =torch.zeros(pair_cnt,2), torch.zeros(pair_cnt,2)
    pre_crowdflow_cnt  = pre_dict['first_frame']
    gt_crowdflow_cnt =  gt_dict['first_frame']
    
    for idx, data in enumerate(zip(pre_dict['inflow'],  pre_dict['outflow'],\
                                   gt_dict['inflow'], gt_dict['outflow']),0):

        inflow_cnt[idx, 0] = data[0]
        inflow_cnt[idx, 1] = data[2]
        outflow_cnt[idx, 0] = data[1]
        outflow_cnt[idx, 1] = data[3]

        if idx %intervals == 0 or  idx== len(pre_dict['inflow'])-1:
            pre_crowdflow_cnt += data[0]
            gt_crowdflow_cnt += data[2]

    return pre_crowdflow_cnt, gt_crowdflow_cnt,  inflow_cnt, outflow_cnt
# def compute_metrics_single_scene(pre_dict, intervals):
#     pair_cnt = len(pre_dict['inflow'])
#     inflow_cnt, outflow_cnt =torch.zeros(pair_cnt,2), torch.zeros(pair_cnt,2)
#     pre_crowdflow_cnt  = pre_dict['first_frame']

#     for idx, data in enumerate(zip(pre_dict['inflow'],  pre_dict['outflow']),0):
#         inflow_cnt[idx, 0] = data[0]
#         outflow_cnt[idx, 0] = data[1]
#         if idx %intervals == 0 or  idx== len(pre_dict['inflow'])-1:
#             pre_crowdflow_cnt += data[0]


#     return pre_crowdflow_cnt,  inflow_cnt, outflow_cnt

# +
def compute_metrics_all_scenes(scenes_pred_dict, scene_gt_dict, intervals):
    scene_cnt = len(scenes_pred_dict)
    metrics = {'MAE':torch.zeros(scene_cnt,2), 'WRAE':torch.zeros(scene_cnt,2), 'MIAE':torch.zeros(0), 'MOAE':torch.zeros(0)}
    for i,(pre_dict, gt_dict) in enumerate( zip(scenes_pred_dict, scene_gt_dict),0):
        time = pre_dict['time']

        pre_crowdflow_cnt, gt_crowdflow_cnt, inflow_cnt, outflow_cnt=\
            compute_metrics_single_scene(pre_dict, gt_dict,intervals)
        print(pre_crowdflow_cnt)
        print(gt_crowdflow_cnt)
        mae = np.abs(pre_crowdflow_cnt - gt_crowdflow_cnt)
        metrics['MAE'][i, :] = torch.tensor([pre_crowdflow_cnt, gt_crowdflow_cnt])
        metrics['WRAE'][i,:] = torch.tensor([mae/(gt_crowdflow_cnt+1e-10), time])

        metrics['MIAE'] =  torch.cat([metrics['MIAE'], torch.abs(inflow_cnt[:,0]-inflow_cnt[:,1])])
        metrics['MOAE'] = torch.cat([metrics['MOAE'], torch.abs(outflow_cnt[:, 0] - outflow_cnt[:, 1])])

    MAE = torch.mean(torch.abs(metrics['MAE'][:, 0] - metrics['MAE'][:, 1]))
    MSE = torch.mean((metrics['MAE'][:, 0] - metrics['MAE'][:, 1]) ** 2).sqrt()
    WRAE = torch.sum(metrics['WRAE'][:,0]*(metrics['WRAE'][:,1]/(metrics['WRAE'][:,1].sum()+1e-10)))*100
    MIAE = torch.mean(metrics['MIAE'] )
    MOAE = torch.mean(metrics['MOAE'])

    return MAE,MSE, WRAE,MIAE,MOAE,metrics['MAE']
# def compute_metrics_all_scenes(scenes_pred_dict, scene_gt_dict, intervals):
#     scene_cnt = len(scenes_pred_dict)
#     metrics = {'MAE':torch.zeros(scene_cnt,2), 'WRAE':torch.zeros(scene_cnt,2)}
#     for i,(pre_dict, gt_dict) in enumerate( zip(scenes_pred_dict, scene_gt_dict),0):
#         time = pre_dict['time']
#         gt_crowdflow_cnt = gt_dict
#         pre_crowdflow_cnt, inflow_cnt, outflow_cnt=\
#             compute_metrics_single_scene(pre_dict,intervals)
#         mae = np.abs(pre_crowdflow_cnt-gt_crowdflow_cnt)

#         metrics['MAE'][i,:] = torch.tensor([pre_crowdflow_cnt, gt_crowdflow_cnt])
#         metrics['WRAE'][i,:] = torch.tensor([mae/(gt_crowdflow_cnt+1e-10), time])

#     MAE =  torch.mean(torch.abs(metrics['MAE'][:,0] - metrics['MAE'][:,1]))
#     MSE = torch.mean((metrics['MAE'][:, 0] - metrics['MAE'][:, 1])**2).sqrt()
#     WRAE = torch.sum(metrics['WRAE'][:,0]*(metrics['WRAE'][:,1]/(metrics['WRAE'][:,1].sum()+1e-10)))*100


#     return MAE, MSE, WRAE, metrics['MAE']
# -

if __name__=='__main__':
    import os
    import random
    import numpy as np
    import torch
    import datasets
    from config import cfg
    from importlib import import_module
    # ------------prepare enviroment------------
    seed = cfg.SEED
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_ID
    torch.backends.cudnn.benchmark = True

    # ------------prepare data loader------------
    data_mode = cfg.DATASET
    datasetting = import_module(f'datasets.setting.{data_mode}')
    cfg_data = datasetting.cfg_data

    # ------------Start Training------------
    pwd = os.path.split(os.path.realpath(__file__))[0]
    cc_trainer = Trainer(cfg_data, pwd)
    cc_trainer.forward()

