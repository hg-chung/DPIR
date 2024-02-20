import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2

def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['train'] # data type
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    i_split=[0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir+ frame['file_path'] + '.png') 
            imgs.append(imageio.imread(fname)) 
            poses.append(np.array(frame['transform_matrix'])) 
        imgs = (np.array(imgs[:200]) / 255.).astype(np.float32) 
        poses = np.array(poses[:200]).astype(np.float32) 
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    print("imgs shape", imgs.shape)
        
    return imgs, poses, [H, W, focal], i_split


