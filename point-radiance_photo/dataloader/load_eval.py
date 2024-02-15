import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2
import scipy.io as sio

def read_mat_file(filename):
    """
    :return: Normal_ground truth in shape: (height, width, 3)
    """
    mat = sio.loadmat(filename)
    gt_n = mat['Normal_gt']
    return gt_n.astype(np.float32)

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['test'] # data type
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    all_normals = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []

        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip   
             
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png') 
            imgs.append(imageio.imread(fname)) 
            poses.append(np.array(frame['transform_matrix'])) 
        imgs = (np.array(imgs[:100]) / 255.).astype(np.float32) 
        poses = np.array(poses[:100]).astype(np.float32) 
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    for i in range(100):
        normal = imageio.imread(os.path.join(basedir,"normal/r_{}_normal.png".format(i+200)))/255
        all_normals.append(2*normal-1)   
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(1)]  
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    mask_files = os.path.join("./data/ball", "mask.png")
    ball_mask = cv2.imread(mask_files, 0).astype(np.float32) / 255.
    gt_normal_files = os.path.join("./data/ball", "Normal_gt.mat")
    ball_normal = read_mat_file(gt_normal_files)      
    print("imgs shape", imgs.shape)
        
    return imgs, poses, all_normals, [H, W, focal], i_split, ball_mask, ball_normal


