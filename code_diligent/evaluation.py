
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "3"
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image

from pyhocon import ConfigFactory
from modules.model_eval import CoreModel
from modules.utils import device, mse2psnr,PSNR, SSIM, LPIPS, \
    grad_loss, safe_path, set_seed, mae_error
from modules.config import config_parser
from dataloader.dataset_eval import Eval_Data
from pyhocon import ConfigFactory


class Evaluater(object):
    def __init__(self, args):
        self.args = args
        self.dataname = args.dataname
        self.logpath = args.basedir
        conf = ConfigFactory.parse_file(args.conf)
        self.conf_model = conf.get_config('model')
        self.splatting_r = self.conf_model.get_float('splatting_r')
        self.data_r = self.conf_model.get_float('data_r')
        self.outpath = safe_path(os.path.join(self.logpath, 'output'))
        self.evalpath = safe_path(os.path.join(self.logpath, 'eval'))
        self.imgpath = safe_path(os.path.join(self.outpath, 'images'))
        self.imgpath = safe_path(os.path.join(self.imgpath, '{}'.format(self.dataname)))
        self.evalpath = safe_path(os.path.join(self.evalpath, '{}'.format(self.dataname)))
        self.imgout_path = safe_path(os.path.join(self.imgpath,
                        'v2_{:.3f}_{:.3f}'.format(self.data_r, self.splatting_r)))
        self.evalout_path = safe_path(os.path.join(self.evalpath,
                        'v2_{:.3f}_{:.3f}'.format(self.data_r, self.splatting_r)))
        for i in [3,7,11,15,19]:
            safe_path(os.path.join(self.evalout_path,"view_{:02d}".format(i+1)))
            
        self.logfile = os.path.join(self.evalout_path, 'log_{}.txt'.format(self.dataname))
        self.weightpath = safe_path(os.path.join(self.imgout_path, 'weight'))
        self.logfile = open(self.logfile, 'w')
        self.loss_fn = torch.nn.MSELoss()
        self.lpips = LPIPS()
        saved_model_state = torch.load(os.path.join(self.weightpath,'model_{}.pth'.format(self.dataname)))
        n_points = saved_model_state["vertsparam"].shape[0]
        self.model = CoreModel(args,n_points).to(device)
        self.imagesgt = torch.tensor(self.model.imagesgt).float().detach().cpu()
        self.masks = torch.tensor(self.model.masks).float().detach().cpu()
        self.normalgt = torch.tensor(self.model.normal).float().detach().cpu()
        self.imagesgt_train = self.imagesgt
        self.training_time = 0
        self.n_train  = self.imagesgt.shape[0]

        self.model.load_state_dict(saved_model_state)
        print(self.imgout_path)
        
    def evaluate(self):
        plt.cla()
        plt.clf()
        loss_all, psnr_all,ssim_all, lpips_all,mae_all = [], [], [], [], []
        self.model.eval()
        for id in range(self.n_train):
            images, normal_img, albedo_img, spec_img, pred_mask, shadow, depth, output= self.model(id,True)
            pred = images[0,...,:3].detach().cpu()
            normal_img = normal_img[0].detach().cpu()
            albedo_img = albedo_img[0].detach().cpu()
            spec_img = spec_img[0].detach().cpu()
            pred_mask = pred_mask[0].detach().cpu().data.numpy()
            mask = self.masks[id].cpu().data.numpy()
            pred_mask[pred_mask>0]=1
            pred_mask[pred_mask==0]=0
            normal_img = normal_img * pred_mask
            mask2 = mask * pred_mask
            pred = pred * mask2 +(1-mask2)
            gt_img = self.imagesgt_train[id]* mask +(1-mask)
            mae = mae_error(normal_img,self.normalgt[id],torch.tensor(mask2).cpu())
            normal_img2 = (normal_img*0.5 +0.5)
            loss = self.loss_fn(pred, gt_img)
            loss_all.append(loss)
            mae_all.append(mae)
            pred = pred.detach().cpu().data.numpy()
            vis_img = shadow[0].expand(normal_img2.shape).detach().cpu().data.numpy().squeeze()
            normal_img2 = normal_img2.detach().cpu().data.numpy().squeeze()
            normal_img = normal_img.detach().cpu().data.numpy().squeeze()
            gt = gt_img.detach().cpu().data.numpy()
            albedo_img = albedo_img.data.numpy().squeeze()
            spec_img = spec_img.data.numpy().squeeze()
            ssim = torch.from_numpy(np.array(SSIM(pred, gt)))
            lpips = torch.from_numpy(np.array(self.lpips(pred, gt)))
            psnr_all.append(torch.from_numpy(np.array(PSNR(pred, gt))))
            ssim_all.append(ssim)
            lpips_all.append(lpips)
            normal_img2 = normal_img2 * mask2 + (1 - mask2)
            albedo_img = albedo_img * mask2 + (1-mask2)
            spec_img = spec_img * mask2 +(1-mask2)
            img = np.concatenate((pred,normal_img2,albedo_img, spec_img,vis_img, gt),1)
            i = (id//96+1)*4
            img = Image.fromarray((img*255).astype(np.uint8))
            img.save(os.path.join(self.evalout_path,"view_{:02d}".format(i),
                'img_{:02d}.png'.format(id%96+1)))
            
            torch.cuda.empty_cache()
        loss_e = torch.stack(loss_all).mean().item()
        psnr_e = torch.stack(psnr_all).mean().item()
        ssim_e = torch.stack(ssim_all).mean().item()
        lpips_e = torch.stack(lpips_all).mean().item()
        mae_e = torch.stack(mae_all).mean().item()
        info = '-----eval-----  loss:{:.3f} psnr:{:.3f} ssim:{:.3f} lpips: {:.3f} mae:{:.3f}'.format(loss_e, psnr_e, ssim_e, lpips_e,mae_e)
        print(info)
        self.logfile.write(info + '\n')
        self.logfile.flush()
        self.model.train()
        return psnr_e

def solve(args):
    trainer = Evaluater(args)
    trainer.evaluate()
    trainer.logfile.flush()


if __name__ == '__main__':
    set_seed(0)
    parser = config_parser()
    args = parser.parse_args()
    dataset = Eval_Data(args)
    args.memitem = dataset.genpc()
    solve(args)
