import os
#os.environ["CUDA_VISIBLE_DEVICES"]= "3"
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image

from pyhocon import ConfigFactory
from modules.model import CoreModel
from modules.utils import device, mse2psnr,\
    grad_loss, safe_path, set_seed
from modules.image_losses import ssim_loss_fn
from modules.config import config_parser
from dataloader.dataset import Data

from pytorch3d.renderer import (
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor
)


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.dataname = args.dataname
        self.logpath = args.basedir
        self.outpath = safe_path(os.path.join(self.logpath, 'output'))
        self.imgpath = safe_path(os.path.join(self.outpath, 'images'))
        self.imgpath = safe_path(os.path.join(self.imgpath, '{}'.format(self.dataname)))
        self.model = CoreModel(args).to(device)
        self.loss_fn = torch.nn.MSELoss()
        self.lr1, self.lr2  = args.lr1, args.lr2
        self.lrexp, self.lr_s = args.lrexp, args.lr_s
        self.set_optimizer(self.lr1, self.lr2)
        self.r_patch = args.r_patch
        conf = ConfigFactory.parse_file(args.conf)
        self.conf_model = conf.get_config('model')
        self.splatting_r = self.conf_model.get_float('splatting_r')
        self.data_r = self.conf_model.get_float('data_r')
        self.conf = conf.get_config('loss')
        self.mse_weight = self.conf.get_float('mse_weight')
        self.ssim_weight = self.conf.get_float('ssim_weight')
        self.sdf_weight = self.conf.get_float('sdf_weight')
        self.mask_weight = self.conf.get_float('mask_weight')
        self.coeff_weight = self.conf.get_float('coeff_weight')
        self.imagesgt = torch.tensor(self.model.imagesgt).float().to(device)
        self.masks = torch.tensor(self.model.masks).float().to(device)
        self.imagesgt_train = self.imagesgt
        self.imgout_path = safe_path(os.path.join(self.imgpath,
                        'v2_{:.3f}_{:.3f}'.format(self.data_r, self.splatting_r)))
        self.weightpath = safe_path(os.path.join(self.imgout_path, 'weight'))
        self.logfile = os.path.join(self.imgout_path, 'log_{}.txt'.format(self.dataname))
        self.logfile = open(self.logfile, 'w')
        self.training_time = 0
        self.n_train = self.imagesgt.shape[0]
        print(self.imgout_path)

    def set_radius(self):
        self.model.raster_settings = PointsRasterizationSettings( # points rasterization params --> 
            bin_size=30, # size of bins for coarse to fine rasterization
            image_size=self.model.img_s, # img size
            radius=self.model.point_radius, # radius 
            points_per_pixel=self.model.raster_n, # number of points to keep track of per pixel
        )
        self.mean_radius = torch.mean(self.model.point_radius)
        
    def set_optimizer(self, lr1=1e-4, lr2=1e-4): # set optimizer
        model_list = [name for name, params in self.model.named_parameters() if 'network' in name]
        model_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in model_list,
                                self.model.named_parameters()))))
        other_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in model_list,
                                self.model.named_parameters()))))
        optimizer = torch.optim.Adam([
            {'params': other_params, 'lr': lr1},
            {'params': model_params, 'lr': lr2}
            ])
        
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.lrexp, -1)
        self.optimizer, self.lr_scheduler = optimizer, lr_scheduler

    def train(self,epoch_n=40, vis_train = False):
        self.logfile.write('-----------Stage Segmentation Line-----------')
        self.logfile.flush()
        max_psnr = 0.
        start_time = time.time()
        for epoch in range(epoch_n):
            loss_all, psnr_all, sdf_all, mask_all, ssim_all, coeff_all= [], [], [], [], [], []
            ids = np.random.permutation(self.n_train)
            for id in tqdm(ids): # train
                self.set_radius()
                images, normal_img, albedo_img, spec_img, pred_mask, output = self.model(id,vis_train) 
                psnr_loss = self.loss_fn(images[0], self.imagesgt_train[id]) 
                pred_img = images.permute(0, 3, 1, 2)
                gt_img = self.imagesgt_train[id].permute(2, 0, 1).unsqueeze(0)
                img_ssim_loss = ssim_loss_fn(pred_img, gt_img, self.masks[id].permute(2,0,1).unsqueeze(0))
                loss = 0
                loss += self.mse_weight * psnr_loss
                loss += self.ssim_weight * img_ssim_loss
                psnr_all.append(mse2psnr(psnr_loss)) # update psnr
                mask_loss = self.loss_fn(pred_mask.reshape(-1).float(), self.masks[id].reshape(-1).float())
                sdf_loss = torch.mean(output['sdf_values'] *output['sdf_values'])
                coeff_loss = output['spec_coeff']
                loss += self.sdf_weight * sdf_loss
                loss += self.mask_weight * mask_loss
                loss += self.coeff_weight * coeff_loss 
                self.optimizer.zero_grad()
                loss.backward() # backward
                self.optimizer.step() # update
                psnr_all.append(mse2psnr(psnr_loss))
                ssim_all.append(img_ssim_loss)
                loss_all.append(loss) 
                sdf_all.append(sdf_loss)
                mask_all.append(mask_loss)
                coeff_all.append(coeff_loss)
                if epoch == epoch_n-1: # visual
                    pred = images[0, ..., :3].detach().cpu().data.numpy()
                    vis_img = pred_mask.expand(normal_img.shape).detach().cpu().data.numpy().squeeze()
                    normal_img = normal_img.detach().cpu().data.numpy().squeeze()
                    albedo_img = albedo_img.detach().cpu().data.numpy().squeeze()
                    spec_img = spec_img.detach().cpu().data.numpy().squeeze()
                    gt = self.imagesgt[id].detach().cpu().data.numpy()
                    # set background as white for visualization
                    mask = self.masks[id].cpu().data.numpy()
                    pred = pred * mask +(1-mask)
                    gt = gt* mask +(1-mask)
                    normal_img = (normal_img*0.5 +0.5)* mask +(1-mask)
                    albedo_img = albedo_img * mask +(1-mask)
                    spec_img = spec_img * mask +(1-mask)
                    img_gt = np.concatenate((pred,normal_img,albedo_img, spec_img, vis_img, gt),1)
                    img_gt = Image.fromarray((img_gt*255).astype(np.uint8))
                    img_gt.save(os.path.join(self.imgout_path,
                            'img_{}_{}_{:.2f}.png'.format(self.dataname, id, mse2psnr(psnr_loss).item())))                    
            self.lr_scheduler.step() # 
            loss_e = torch.stack(loss_all).mean().item() 
            psnr_e = torch.stack(psnr_all).mean().item() 
            sdf_e = torch.stack(sdf_all).mean().item() 
            mask_e = torch.stack(mask_all).mean().item() 
            coeff_e = torch.stack(coeff_all).mean().item() 
            ssim_e = torch.stack(ssim_all).mean().item()
            info = '-----train-----  epoch:{} loss:{:.3f} psnr:{:.3f} ssimloss:{:.3f} sdf_loss:{:.3f} mask_loss:{:.3f} coeff_loss:{:.3f} number:{} radius:{:.5f} lr1:{} lr2:{} ' \
                                        .format(epoch, loss_e, psnr_e, ssim_e, sdf_e, mask_e, coeff_e, self.model.vertsparam.shape[0],self.mean_radius, \
                                                self.lr_scheduler.get_last_lr()[0],self.lr_scheduler.get_last_lr()[1])
            print(info)
            self.logfile.write(info + '\n')
            self.logfile.flush()
        
        self.training_time += time.time()-start_time
        torch.save(self.model.state_dict(), os.path.join(
                 self.weightpath,'model_{}.pth'.format(self.dataname)))

def solve(args):
    trainer = Trainer(args)
    trainer.train()
    for i in range(args.refine_n):
        trainer.model.remove_out()
        trainer.model.repeat_pts()
        trainer.set_optimizer(args.lr1, args.lr2)
        trainer.train()
    trainer.logfile.write('Total Training Time: '
                  '{:.2f}s\n'.format(trainer.training_time))
    trainer.logfile.flush()  
    print('Training time: {:.2f} s'.format(trainer.training_time))
    

if __name__ == '__main__':
    set_seed(0)
    parser = config_parser()
    args = parser.parse_args()
    dataset = Data(args)
    args.memitem = dataset.genpc() 
    solve(args)
