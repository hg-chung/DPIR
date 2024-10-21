import os
#os.environ["CUDA_VISIBLE_DEVICES"]= "0"
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image

from pyhocon import ConfigFactory
from modules.model_eval import CoreModel
from modules.utils import device, mse2psnr, SSIM,LPIPS, safe_path, set_seed, mae_error
from modules.image_losses import ssim_loss_fn
from modules.config import config_parser
from dataloader.dataset_eval import Eval_Data
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor
)


class Evaluater(object):
    def __init__(self, args):
        self.args = args
        self.dataname = args.dataname
        self.logpath = args.basedir
        conf = ConfigFactory.parse_file(args.conf)
        self.conf = conf.get_config('model')
        self.splatting_r = self.conf.get_float('splatting_r')
        self.data_r = self.conf.get_float('data_r')
        self.outpath = safe_path(os.path.join(self.logpath, 'output'))
        self.imgpath = safe_path(os.path.join(self.outpath, 'images'))
        self.imgpath = safe_path(os.path.join(self.imgpath, '{}'.format(self.dataname)))
        self.evalpath = safe_path(os.path.join(self.logpath, 'eval'))
        self.evalimg_path = safe_path(os.path.join(self.evalpath, '{}'.format(self.dataname)))
        self.imgout_path = safe_path(os.path.join(self.imgpath,
                        'v2_{:.3f}_{:.3f}'.format(self.data_r, self.splatting_r)))
        self.weightpath = safe_path(os.path.join(self.imgout_path, 'weight'))
        self.evalout_path = safe_path(os.path.join(self.evalimg_path,
                        'v2_{:.3f}_{:.3f}'.format(self.data_r, self.splatting_r)))
        self.loss_fn = torch.nn.MSELoss()
        self.lpips = LPIPS()
        self.lrexp, self.lr_s = args.lrexp, args.lr_s
        saved_model_state = torch.load(os.path.join(self.weightpath,'model_{}.pth'.format(self.dataname)))
        n_points = saved_model_state["vertsparam"].shape[0]
        print(n_points)
        self.model = CoreModel(args,n_points).to(device)
        self.model.load_state_dict(saved_model_state)
        self.r_patch = args.r_patch
        
        self.imagesgt = torch.tensor(self.model.imagesgt).float().detach().cpu()
        self.masks = torch.tensor(self.model.masks).float().detach().cpu()
        self.imagesgt_train = self.imagesgt
        self.gt_normals = torch.tensor(self.model.normals).float().detach().cpu()
        self.logfile = os.path.join(self.evalout_path, 'log_{}.txt'.format(self.dataname))
        self.logfile = open(self.logfile, 'w')
        self.training_time = 0
        self.n_train = self.imagesgt.shape[0]
        print(self.evalout_path)

    def evaluate(self,start=0, end=100, vis_train =False):
        loss_all, psnr_all,ssim_all, lpips_all, mae_all = [], [], [], [],[]
        for id in range(start,end): # train
            images, normal_img, albedo_img, spec_img, pred_mask, depth, output = self.model(id,vis_train) 
            pred = images[0,...,:3].detach().cpu()
            normal_img = normal_img[0].detach().cpu() 
            albedo_img = albedo_img[0].detach().cpu()
            spec_img = spec_img[0].detach().cpu()
            pred_mask = pred_mask[0].detach().cpu()
            depth_img = depth[0].detach().cpu()
            pred_mask[pred_mask>0.5] = 1
            pred_mask[pred_mask<=0.5] = 0
            mask = self.masks[id].cpu()
            mask[mask>0.5] = 1
            mask[mask<=0.5] = 0
            mask2 = mask *pred_mask
            normal_img = normal_img * pred_mask
            mae = mae_error(normal_img*mask2,self.gt_normals[id]*mask,torch.tensor(mask2).cpu())
            mae_all.append(mae)
            psnr_loss = self.loss_fn(pred, self.imagesgt_train[id])
            psnr_all.append(mse2psnr(psnr_loss)) 
            pred = images[0, ..., :3].detach().cpu().data.numpy()
            vis_img = pred_mask.expand(normal_img.shape).detach().cpu().data.numpy().squeeze()
            depth_img = depth_img.expand(normal_img.shape).detach().cpu().data.numpy().squeeze()
            normal_img = normal_img.detach().cpu().data.numpy().squeeze()
            albedo_img = albedo_img.detach().cpu().data.numpy().squeeze()
            spec_img = spec_img.detach().cpu().data.numpy().squeeze()
            gt = self.imagesgt[id].detach().cpu().data.numpy()
            ssim = torch.from_numpy(np.array(SSIM(pred, gt)))
            lpips = torch.from_numpy(np.array(self.lpips(pred, gt)))
            ssim_all.append(ssim)
            lpips_all.append(lpips)
            # set background as white for visualization
            mask2 = mask2.data.numpy()
            pred = pred * mask2+(1-mask2)
            gt = gt* mask2 +(1-mask2)
            normal_img = (normal_img*0.5 +0.5)* mask2 +(1-mask2)
            albedo_img = albedo_img * mask2 +(1-mask2)
            spec_img = spec_img * mask2 +(1-mask2)
            vis_img = vis_img * mask2
            depth_img = depth_img * mask2
            img_gt = np.concatenate((pred,normal_img,albedo_img, spec_img,vis_img,depth_img, gt),1)    
            img_gt = Image.fromarray((img_gt*255).astype(np.uint8))
            img_gt.save(os.path.join(self.evalout_path,
                    'img_{}_{}.png'.format(self.dataname, id)))                  
        psnr_e = torch.stack(psnr_all).mean().item()
        ssim_e = torch.stack(ssim_all).mean().item()
        lpips_e = torch.stack(lpips_all).mean().item()
        mae_e = torch.stack(mae_all).mean().item()
        info = '-----eval----- psnr:{:.3f} ssim:{:.3f} lpips: {:.3f} mae: {:.3f}'.format( psnr_e, ssim_e, lpips_e, mae_e)
        print(info)
        self.logfile.write(info + '\n')
        self.logfile.flush()

def solve(args):
    trainer = Evaluater(args)
    trainer.evaluate()
    trainer.logfile.flush()
    print('Training time: {:.2f} s'.format(trainer.training_time))
    

if __name__ == '__main__':
    set_seed(0)
    parser = config_parser()
    args = parser.parse_args()
    dataset = Eval_Data(args)
    args.memitem = dataset.genpc() 
    solve(args)
