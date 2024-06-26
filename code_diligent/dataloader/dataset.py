
import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataloader.load_diligent import load_diligent_data

def safe_path(path):
    if os.path.exists(path):
        return path
    else:
        os.mkdir(path)
        return path


# Ray helpers
def get_rays(H, W, K, c2w):
    device = c2w.device

    i, j = torch.meshgrid(torch.linspace(0, W-1, W, device=device),
                          torch.linspace(0, H-1, H, device=device))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i, device=device)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


# Ray helpers
def get_uvs_from_ray(H, W, K, c2w,pts):
    RP = torch.bmm(c2w[:3,:3].T[None,:,:].repeat(pts.shape[0],1,1),pts[:,:,None])[:,:,0]
    t = torch.mm(c2w[:3,:3].T,-c2w[:3,-1][:,None])
    pts_local0 = torch.sum((pts-c2w[:3,-1])[..., None, :] * (c2w[:3,:3].T), -1)
    pts_local = pts_local0/(-pts_local0[...,-1][...,None]+1e-7)
    u = pts_local[...,0]*K[0][0]+K[0][2]
    v = -pts_local[...,1]*K[1][1]+K[1][2]
    uv = torch.stack((u,v),-1)
    return uv,pts_local0


def batch_get_uv_from_ray(H,W,K,poses,pts):
    RT = (poses[:, :3, :3].transpose(1, 2))
    pts_local = torch.sum((pts[..., None, :] - poses[:, :3, -1])[..., None, :] * RT, -1)
    pts_local = pts_local / (-pts_local[..., -1][..., None] + 1e-7)
    u = pts_local[..., 0] * K[0][0] + K[0][2]
    v = -pts_local[..., 1] * K[1][1] + K[1][2]
    uv0 = torch.stack((u, v), -1)
    uv0[...,0] = uv0[...,0]/W*2-1
    uv0[...,1] = uv0[...,1]/H*2-1
    uv0 = uv0.permute(2,0,1,3)
    return uv0


class MemDataset(object):
    def __init__(self,pts,pose,image,mask,K,light_direction,light_intensity):
        self.pts = pts
        self.pose = pose
        self.mask = mask
        self.image = image
        self.K = K
        self.light_direction = light_direction
        self.light_intensity = light_intensity

class test_MemDataset(object):
    def __init__(self,pose,image,mask,K,light_direction,light_intensity):
        self.pose = pose
        self.mask = mask
        self.image = image
        self.K = K
        self.light_direction = light_direction
        self.light_intensity = light_intensity
        
class Data():
    def __init__(self, args):
        self.dataname = args.dataname
        self.datadir = os.path.join(args.datadir,args.dataname)
        self.logpath = safe_path(args.basedir)
        images, masks, poses, light_dir, light_int, hwf, K, i_split, mask_imgs, mask_poses = load_diligent_data(self.datadir)     
           
        self.i_split = i_split
        self.images = images
        self.masks = masks
        self.poses = poses
        self.light_direction = light_dir
        self.light_intensity = light_int
        self.mask_imgs = mask_imgs
        self.mask_poses = mask_poses
        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]   
        print(K)
        self.K = K
        self.hwf = hwf

    def genpc(self):
        
        [H, W, focal] = self.hwf
        K = torch.tensor(self.K)
        train_n = self.mask_imgs.shape[0]
        poses = torch.tensor(self.mask_poses)[:train_n]
        images = torch.tensor(self.mask_imgs)[:train_n] 
        images= torch.cat([images]*3,dim=-1)
        print("image shape",images.shape)
        
        pc,color,N = [],[], H
        [xs,ys,zs],[xe,ye,ze] = [-2,-2,-2],[2,2,2]
        pts_all = []
        for h_id in tqdm(range(N)):
            i, j = torch.meshgrid(torch.linspace(xs, xe, N),
                                  torch.linspace(ys, ye, N)) 
            i, j = i.t(), j.t()
            pts = torch.stack([i, j, torch.ones_like(i)], -1)
            pts[...,2] = h_id / N * (ze - zs) + zs
            pts_all.append(pts.clone())
            uv = batch_get_uv_from_ray(H,W,K,poses,pts)
            result = F.grid_sample(images.permute(0, 3, 1, 2).float(), uv).permute(0,2,3,1)

            margin = 0.05
            result[(uv[..., 0] >= 1.0) * (uv[..., 0] <= 1.0 + margin)] = 1
            result[(uv[..., 0] >= -1.0 - margin) * (uv[..., 0] <= -1.0)] = 1
            result[(uv[..., 1] >= 1.0) * (uv[..., 1] <= 1.0 + margin)] = 1
            result[(uv[..., 1] >= -1.0 - margin) * (uv[..., 1] <= -1.0)] = 1
            result[(uv[..., 0] <= -1.0 - margin) + (uv[..., 0] >= 1.0 + margin)] = 0
            result[(uv[..., 1] <= -1.0 - margin) + (uv[..., 1] >= 1.0 + margin)] = 0

            img = ((result>0.).sum(0)[...,0]>train_n-1).float()
            pc.append(img)
            color.append(result.mean(0))
        pc = torch.stack(pc,-1)
        color = torch.stack(color,-1)
        r, g, b = color[:, :, 0], color[:, :, 1], color[:, :, 2]
        idx = torch.where(pc > 0)
        color = torch.stack((r[idx],g[idx],b[idx]),-1)
        idx = (idx[1],idx[0],idx[2])
        pts = torch.stack(idx,-1).float()/N
        pts[:,0] = pts[:,0]*(xe-xs)+xs
        pts[:,1] = pts[:,1]*(ye-ys)+ys
        pts[:,2] = pts[:,2]*(ze-zs)+zs

        pts = torch.cat((pts,color),-1).cpu().data.numpy()
        print('Initialization, data:{} point:{}'.format(self.dataname,pts.shape))
        item = MemDataset(pts,self.poses,self.images,self.masks,self.K, self.light_direction, self.light_intensity)
        return item
