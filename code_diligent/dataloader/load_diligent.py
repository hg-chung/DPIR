import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2

def load_diligent_data(basedir):
    with open(os.path.join(basedir, 'params.json'), 'r') as fp:
        meta = json.load(fp)
    
    all_imgs = []
    masks =[]
    poses=[]
    all_poses =[]
    all_light_dir =[]
    all_light_int =[]
    mask_imgs=[]
    mask_poses=[]
    
    with open(os.path.join(basedir, 'params.json'), 'r') as fp:
        meta = json.load(fp)

    v_train = meta['view_train'] # 15 views
    i_split=[0]
    #l_train = [5,8,19,23,29,32,42,47,50,56,63,67,77,88,91,95] # ps-nerf 16 lights
    l_train = [25, 30, 35, 42, 46, 52, 62, 67, 72, 81, 86, 90 ] # 16 lights for bear due to saturation problem
    #l_train = [5,29,50,77] # 4 lightings
    #l_train = [5,19,29,32,42,50,63,77,88,91] # 10 lightings
    #l_train = [1,5,8,10,13,16,19,23,25,29,32,36,42,45,47,50,52,56,60,63,67,70,72,75,77,83,88,91,92,95] # 30 lightings
    
    KK = np.array(meta['K']).astype(np.float32) #intrinsic parameters
    light_direction = np.array(meta['light_direction']).astype(np.float32)
    focal = KK[0][0]
    poses = np.array(meta['pose_c2w']).astype(np.float32) 

    for vi in v_train:
        mask = imageio.imread(os.path.join(basedir,"./mask/view_{:02d}.png".format(vi+1)))
        H, W = mask.shape[0],mask.shape[1]
        mask = mask.reshape(H,W,1)
        mask = mask/ 255.
        mask_imgs.append(mask)
        mask_poses.append(poses[vi])

    for vi in v_train:
        for li in l_train:
            mask = imageio.imread(os.path.join(basedir,"./mask/view_{:02d}.png".format(vi+1)))
            mask = mask.reshape(H,W,1)
            mask = mask/ 255.
            img = imageio.imread(os.path.join(basedir,"./img_intnorm_gt/view_{:02d}/{:03d}.png".format(vi+1,li+1)))
            light_dir = np.einsum('ij,kj->ki',poses[vi,:3,:3],light_direction[li].reshape(1,3))
            light_int = np.ones((3,)) *0.2 # normalized and scaled factor
            all_light_dir.append(light_dir)
            all_light_int.append(light_int)
            all_imgs.append(img*mask)
            masks.append(mask)
            all_poses.append(poses[vi])
     
    mask_imgs = (np.array(mask_imgs)).astype(np.float32)  
    mask_poses = np.array(mask_poses).astype(np.float32)
    imgs = (np.array(all_imgs)/ 255.).astype(np.float32)   
    masks = np.array(masks).astype(np.float32)
    all_poses = np.array(all_poses).astype(np.float32)
    all_light_dir = np.array(all_light_dir).astype(np.float32)
    all_light_int = np.array(all_light_int).astype(np.float32)
    print("mask_imgs shape", mask_imgs.shape)
    print("mask_poses shape", mask_poses.shape)
    print("imgs shape", imgs.shape)
    print("poses shape", all_poses.shape)
    print("light dir shape",all_light_dir.shape)
    print("light int shape",all_light_int.shape)
    

        
    return imgs, masks, all_poses, all_light_dir, all_light_int, [H,W,focal], KK, i_split , mask_imgs, mask_poses

