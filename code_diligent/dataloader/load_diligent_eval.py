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

def load_diligent_relight(basedir):
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
    #l_test = [25, 30, 35, 42, 46, 52, 62, 67, 72, 81, 86, 90 ] # bear train idx
    #l_test = [22, 28, 33, 38, 40, 49, 57, 65, 70, 77, 84, 93] # bear eval idx
    l_test = np.arange(0,96)
    KK = np.array(meta['K']).astype(np.float32)
    focal= KK[0][0]
    light_direction = np.array(meta['light_direction']).astype(np.float32)
    poses = np.array(meta['pose_c2w']).astype(np.float32) # all views 20 views
 
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

    mask_files = os.path.join("./data/ball", "mask.png")
    ball_mask = cv2.imread(mask_files, 0).astype(np.float32) / 255.
    gt_normal_files = os.path.join("./data/ball", "Normal_gt.mat")
    ball_normal = read_mat_file(gt_normal_files)      
        
    imgs = (np.array(all_imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)   
    masks = np.array(masks).astype(np.float32)
    all_poses = np.array(all_poses).astype(np.float32)
    all_light_dir = np.array(all_light_dir).astype(np.float32)
    all_light_int = np.array(all_light_int).astype(np.float32)
    all_normal = (np.array(all_normal)).astype(np.float32)
    print("eval_imgs shape", imgs.shape)
    print("eval_poses shape", all_poses.shape)
    print("eval_light dir shape",all_light_dir.shape)
    print("eval_light int shape",all_light_int.shape)
        
    return imgs, masks, all_poses, all_light_dir, all_light_int, all_normal, [H,W,focal], KK, i_split, ball_mask, ball_normal

