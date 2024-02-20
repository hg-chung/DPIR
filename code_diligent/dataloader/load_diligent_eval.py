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

def load_diligent_eval(basedir):
    with open(os.path.join(basedir, 'params.json'), 'r') as fp:
        meta = json.load(fp)
    
    all_imgs = []
    masks =[]
    poses=[]
    all_poses =[]
    all_light_dir =[]
    all_light_int =[]
    all_normal = []
    
    with open(os.path.join(basedir, 'params.json'), 'r') as fp:
        meta = json.load(fp)

    v_test = meta['view_test']
    i_split=[0]
    #l_test = [22, 28, 33, 38, 40, 49, 57, 65, 70, 77, 84, 93] # bear eval idx
    l_test = np.arange(0,96)
    KK = np.array(meta['K']).astype(np.float32)
    focal= KK[0][0]
    light_direction = np.array(meta['light_direction']).astype(np.float32)
    poses = np.array(meta['pose_c2w']).astype(np.float32) 
 
    for vi in v_test:
        for li in l_test:
            mask = imageio.imread(os.path.join(basedir,"./mask/view_{:02d}.png".format(vi+1)))
            H, W =  mask.shape[0], mask.shape[1]
            mask = mask.reshape(H,W,1)
            mask = mask/ 255.
            img = imageio.imread(os.path.join(basedir,"./img_intnorm_gt/view_{:02d}/{:03d}.png".format(vi+1,li+1)))
            light_dir = np.einsum('ij,kj->ki',poses[vi,:3,:3],light_direction[li].reshape(1,3))
            light_int = np.ones((3,)) *0.2 # normalized and scale factor
            gt_normal = np.load(os.path.join(basedir,"./normal/npy/view_{:02d}.npy".format(vi+1)))
            all_light_dir.append(light_dir)
            all_light_int.append(light_int)
            all_imgs.append(img*mask)
            masks.append(mask)
            all_poses.append(poses[vi])
            all_normal.append(gt_normal)

    imgs = (np.array(all_imgs) / 255.).astype(np.float32) 
    masks = np.array(masks).astype(np.float32)
    all_poses = np.array(all_poses).astype(np.float32)
    all_light_dir = np.array(all_light_dir).astype(np.float32)
    all_light_int = np.array(all_light_int).astype(np.float32)
    all_normal = (np.array(all_normal)).astype(np.float32)
    print("eval_imgs shape", imgs.shape)
    print("eval_poses shape", all_poses.shape)
    print("eval_light dir shape",all_light_dir.shape)
    print("eval_light int shape",all_light_int.shape)
        
    return imgs, masks, all_poses, all_light_dir, all_light_int, all_normal, [H,W,focal], KK, i_split, 

