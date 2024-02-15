
import os
import torch
import numpy as np
import open3d as o3d
from torch.nn import functional as F
from skimage.metrics import structural_similarity
import lpips
import math
import imageio
from PIL import Image
import cv2

# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

mse2psnr = lambda x : -10. * torch.log(x) \
            / torch.log(torch.tensor([10.], device=x.device))

def rotation_matrix(A,B):
# a and b are in the form of numpy array

   ax = A[0]
   ay = A[1]
   az = A[2]

   bx = B[0]
   by = B[1]
   bz = B[2]

   au = A/(np.sqrt(ax*ax + ay*ay + az*az))
   bu = B/(np.sqrt(bx*bx + by*by + bz*bz))

   R=np.array([[bu[0]*au[0], bu[0]*au[1], bu[0]*au[2]], [bu[1]*au[0], bu[1]*au[1], bu[1]*au[2]], [bu[2]*au[0], bu[2]*au[1], bu[2]*au[2]] ])


   return(R)  
  
def pose_shadow(pose,light_dir):
    z_rot= rotation_matrix(pose[:3,2],light_dir)
    
    initial_y_axis = np.array([0, 1, 20])

    x_axis = np.cross(initial_y_axis, light_dir)
    x_axis/= np.linalg.norm(x_axis)

    y_axis = np.cross(light_dir, x_axis)
    y_axis/= np.linalg.norm(y_axis)

    light_trans = np.dot(z_rot,pose[:3,3]).reshape(3,1)
    shadow_mat = np.concatenate((x_axis.reshape(3,1),y_axis.reshape(3,1),light_dir.reshape(3,1),light_trans),axis=1)
    shadow_mat = np.concatenate((shadow_mat,np.array([0.,0.,0.,1.]).reshape(1,4)),axis=0)
    
    return shadow_mat

def PSNR(img1, img2, mask=None):
    '''
    Input : H x W x 3   [0,1]
    Output : PSNR
    '''
    img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
    if mask is not None:
        img1, img2 = img1[mask.astype(bool)], img2[mask.astype(bool)]
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        psnr = 100
    else:
        psnr = - 10.0 * math.log10(mse)
    return psnr
    
   
def SSIM(img1, img2, mask=None, data_range=1, channel_axis=2, gaussian_weights=True, sigma=1.5,  use_sample_covariance=False):
    '''
    Input : H x W x 3   [0,1]
    Output : SSIM
    '''
    ssim = structural_similarity(img1, img2,
                        data_range=data_range, channel_axis=channel_axis, 
                        gaussian_weights=gaussian_weights, sigma=sigma, 
                        use_sample_covariance=use_sample_covariance)
    return ssim
   
class LPIPS():
    def __init__(self, net='alex'):
        '''
        Input : H x W x 3   [0,1]
        Output : LPIPS
        '''
        self.loss_fn = lpips.LPIPS(net=net).cuda()
        self.loss_fn.eval()

    def __call__(self, img1, img2, mask=None):
        img1 = torch.from_numpy(img1).permute(2,0,1).unsqueeze(0).to(torch.float32).cuda()*2.-1
        img2 = torch.from_numpy(img2).permute(2,0,1).unsqueeze(0).to(torch.float32).cuda()*2.-1
        err = self.loss_fn.forward(img1,img2,normalize=True)
        return err.item()

def mae_error(img1, img2, mask=None, normalize=True):
        if mask is not None:
            img1, img2 = torch.masked_select(img1,mask.to(torch.bool)).view(-1,3), torch.masked_select(img2,mask.to(torch.bool)).view(-1,3)
        if normalize:
            img1 = F.normalize(img1, dim=-1)
            img2 = F.normalize(img2, dim=-1)
        dot_product = (img1 * img2).sum(-1).clamp(-1, 1)
        angular_err = torch.acos(dot_product) * 180.0 / math.pi
        l_err_mean  = angular_err.mean()
        return l_err_mean#, angular_err   

def safe_path(path):
    if os.path.exists(path):
        return path
    else:
        os.mkdir(path)
        return path

def load_mem_data(mem):
    poses = mem.pose # w2c
    R, T1 = (poses[:, :3, :3]), poses[:, :3, -1]
    R, T2 = R, -(T1[: ,None ,:] @ R)[: ,0]
    return mem.pts, mem.image, mem.K, R, T1, T2, poses, mem.mask, mem.light_direction, mem.light_intensity

def load_test_data(mem):
    poses = mem.pose # w2c
    R, T1 = (poses[:, :3, :3]), poses[:, :3, -1]
    R, T2 = R, -(T1[: ,None ,:] @ R)[: ,0]
    return mem.image, mem.K, R, T1, T2, poses, mem.mask, mem.light_direction, mem.light_intensity, mem.normal, mem.ball_mask, mem.ball_normal

def get_rays(H, W, K, c2w):
    device = c2w.device
    i, j = torch.meshgrid(torch.linspace(0, W-1, W, device=device),
            torch.linspace(0, H-1, H, device=device))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0],
            -(j-K[1][2])/K[1][1], -torch.ones_like(i, device=device)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def remove_outlier(pts):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd = pcd.voxel_down_sample(voxel_size=0.010)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return np.array(pcd.points)[np.array(ind)]


def grad_loss(output, gt):
    def one_grad(shift):
        ox = output[shift:] - output[:-shift]
        oy = output[:, shift:] - output[:, :-shift]
        gx = gt[shift:] - gt[:-shift]
        gy = gt[:, shift:] - gt[:, :-shift]
        loss = (ox - gx).abs().mean() + (oy - gy).abs().mean()
        return loss
    loss = (one_grad(1) + one_grad(2) + one_grad(3)) / 3.
    return loss

def get_camera_params(uv, pose, intrinsics):
    if pose.shape[1] == 7: #In case of quaternion vector representation
        cam_loc = pose[:, 4:]
        R = quat_to_rot(pose[:,:4])
        p = torch.eye(4).repeat(pose.shape[0],1,1).cuda().float()
        p[:, :3, :3] = R
        p[:, :3, 3] = cam_loc
    else: # In case of pose matrix representation
        cam_loc = pose[:, :3, 3]
        p = pose

    batch_size, num_samples, _ = uv.shape

    depth = torch.ones((batch_size, num_samples)).cuda()
    x_cam = uv[:, :, 0].view(batch_size, -1)
    y_cam = uv[:, :, 1].view(batch_size, -1)
    z_cam = depth.view(batch_size, -1)

    pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics)

    # permute for batch matrix product
    pixel_points_cam = pixel_points_cam.permute(0, 2, 1)

    world_coords = torch.bmm(p, pixel_points_cam).permute(0, 2, 1)[:, :, :3]
    ray_dirs = world_coords - cam_loc[:, None, :]
    ray_dirs = F.normalize(ray_dirs, dim=2)

    return ray_dirs, cam_loc

def lift(x, y, z, intrinsics):
    # parse intrinsics
    intrinsics = intrinsics.cuda()
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    sk = intrinsics[:, 0, 1]

    x_lift = (x - cx.unsqueeze(-1) + cy.unsqueeze(-1)*sk.unsqueeze(-1)/fy.unsqueeze(-1) - sk.unsqueeze(-1)*y/fy.unsqueeze(-1)) / fx.unsqueeze(-1) * z
    y_lift = (y - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z

    # homogeneous
    return torch.stack((x_lift, y_lift, z, torch.ones_like(z).cuda()), dim=-1)

def quat_to_rot(q):
    batch_size, _ = q.shape
    q = F.normalize(q, dim=1)
    R = torch.ones((batch_size, 3,3)).cuda()
    qr=q[:,0]
    qi = q[:, 1]
    qj = q[:, 2]
    qk = q[:, 3]
    R[:, 0, 0]=1-2 * (qj**2 + qk**2)
    R[:, 0, 1] = 2 * (qj *qi -qk*qr)
    R[:, 0, 2] = 2 * (qi * qk + qr * qj)
    R[:, 1, 0] = 2 * (qj * qi + qk * qr)
    R[:, 1, 1] = 1-2 * (qi**2 + qk**2)
    R[:, 1, 2] = 2*(qj*qk - qi*qr)
    R[:, 2, 0] = 2 * (qk * qi-qj * qr)
    R[:, 2, 1] = 2 * (qj*qk + qi*qr)
    R[:, 2, 2] = 1-2 * (qi**2 + qj**2)
    return R

def set_seed(seed=0):
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_light(path, light_h=None):
    ext = os.path.basename(path).split('.')[-1]
    if ext == 'exr':
        arr = read_exr(path)
    elif ext == 'hdr':
        arr = read_hdr(path)
    else:
        raise NotImplementedError(ext)
    if light_h:
        arr = cv2.resize(arr, (2*light_h, light_h), interpolation=cv2.INTER_LINEAR)
    return arr

def read_exr(path):
    arr = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if arr is None:
        raise RuntimeError(f"Failed to read\n\t{path}")
    # RGB
    if arr.ndim == 3 or arr.shape[2] == 3:
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return rgb
    raise NotImplementedError(arr.shape)

def read_hdr(path):
    with open(path, 'rb') as h:
        buffer_ = np.fromstring(h.read(), np.uint8)
    bgr = cv2.imdecode(buffer_, cv2.IMREAD_UNCHANGED)
    rgb = cv2.cv2tColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def write_hdr(rgb, outpath):
    # Writes a ``float32`` RGB array as an HDR map to disk.
    assert rgb.dtype == np.float32, "Input must be float32"
    os.makedirs(os.path.dirname(outpath),exist_ok=True)

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    success = cv2.imwrite(outpath, bgr)
    assert success, "Writing HDR failed"

def vis_light(light_probe, outpath=None, h=None):
    # In case we are predicting too low of a resolution
    if h is not None:
        light_probe = cv2.resize(light_probe, (2*h, h), interpolation=cv2.INTER_NEAREST)
    # Tonemap
    tonemap = lambda hdr, gamma: (hdr / hdr.max()) ** (1 / gamma)
    img = tonemap(light_probe, gamma=4) # [0, 1]
    img_uint = (img*255).astype(np.uint8)
    # Optionally, write to disk
    if outpath is not None:
        Image.fromarray(img_uint).save(outpath)
    return img_uint


def gen_light_xyz(envmap_h, envmap_w, envmap_radius=1e2):
    """Additionally returns the associated solid angles, for integration.
    """
    # OpenEXR "latlong" format
    # lat = pi/2
    # lng = pi
    #     +--------------------+
    #     |                    |
    #     |                    |
    #     +--------------------+
    #                      lat = -pi/2
    #                      lng = -pi
    lat_step_size = np.pi / (envmap_h + 2)
    lng_step_size = 2 * np.pi / (envmap_w + 2)
    # Try to exclude the problematic polar points
    lats = np.linspace(
        np.pi / 2 - lat_step_size, -np.pi / 2 + lat_step_size, envmap_h)
    lngs = np.linspace(
        np.pi - lng_step_size, -np.pi + lng_step_size, envmap_w)
    lngs, lats = np.meshgrid(lngs, lats)

    # To Cartesian
    rlatlngs = np.dstack((envmap_radius * np.ones_like(lats), lats, lngs))
    rlatlngs = rlatlngs.reshape(-1, 3)
    xyz = sph2cart(rlatlngs)
    xyz = xyz.reshape(envmap_h, envmap_w, 3)

    # Calculate the area of each pixel on the unit sphere (useful for
    # integration over the sphere)
    sin_colat = np.sin(np.pi / 2 - lats)
    areas = 4 * np.pi * sin_colat / np.sum(sin_colat)

    assert 0 not in areas, \
        "There shouldn't be light pixel that doesn't contribute"

    return xyz, areas
def _warn_degree(angles):
    if (np.abs(angles) > 2 * np.pi).any():
        print(
            "Some input value falls outside [-2pi, 2pi]. You sure inputs are "
            "in radians")

def _convert_sph_conventions(pts_r_angle1_angle2, what2what):
    """Internal function converting between different conventions for
    spherical coordinates. See :func:`cart2sph` for conventions.
    """
    if what2what == 'lat-lng_to_theta-phi':
        pts_r_theta_phi = np.zeros(pts_r_angle1_angle2.shape)
        # Radius is the same
        pts_r_theta_phi[:, 0] = pts_r_angle1_angle2[:, 0]
        # Angle 1
        pts_r_theta_phi[:, 1] = np.pi / 2 - pts_r_angle1_angle2[:, 1]
        # Angle 2
        ind = pts_r_angle1_angle2[:, 2] < 0
        pts_r_theta_phi[ind, 2] = 2 * np.pi + pts_r_angle1_angle2[ind, 2]
        pts_r_theta_phi[np.logical_not(ind), 2] = \
            pts_r_angle1_angle2[np.logical_not(ind), 2]
        return pts_r_theta_phi

    if what2what == 'theta-phi_to_lat-lng':
        pts_r_lat_lng = np.zeros(pts_r_angle1_angle2.shape)
        # Radius is the same
        pts_r_lat_lng[:, 0] = pts_r_angle1_angle2[:, 0]
        # Angle 1
        pts_r_lat_lng[:, 1] = np.pi / 2 - pts_r_angle1_angle2[:, 1]
        # Angle 2
        ind = pts_r_angle1_angle2[:, 2] > np.pi
        pts_r_lat_lng[ind, 2] = pts_r_angle1_angle2[ind, 2] - 2 * np.pi
        pts_r_lat_lng[np.logical_not(ind), 2] = \
            pts_r_angle1_angle2[np.logical_not(ind), 2]
        return pts_r_lat_lng

    raise NotImplementedError(what2what)


def uniform_sample_sph(n, r=1, convention='lat-lng'):
    r"""Uniformly samples points on the sphere
    [`source <https://mathworld.wolfram.com/SpherePointPicking.html>`_].

    Args:
        n (int): Total number of points to sample. Must be a square number.
        r (float, optional): Radius of the sphere. Defaults to :math:`1`.
        convention (str, optional): Convention for spherical coordinates.
            See :func:`cart2sph` for conventions.

    Returns:
        numpy.ndarray: Spherical coordinates :math:`(r, \theta_1, \theta_2)`
        in radians. The points are ordered such that all azimuths are looped
        through first at each elevation.
    """
    n_ = np.sqrt(n)
    if n_ != int(n_):
        raise ValueError("%d is not perfect square" % n)
    n_ = int(n_)

    pts_r_theta_phi = []
    for u in np.linspace(0, 1, n_):
        for v in np.linspace(0, 1, n_):
            theta = np.arccos(2 * u - 1) # [0, pi]
            phi = 2 * np.pi * v # [0, 2pi]
            pts_r_theta_phi.append((r, theta, phi))
    pts_r_theta_phi = np.vstack(pts_r_theta_phi)

    # Select output convention
    if convention == 'lat-lng':
        pts_sph = _convert_sph_conventions(
            pts_r_theta_phi, 'theta-phi_to_lat-lng')
    elif convention == 'theta-phi':
        pts_sph = pts_r_theta_phi
    else:
        raise NotImplementedError(convention)

    return pts_sph


def cart2sph(pts_cart, convention='lat-lng'):
    r"""Converts 3D Cartesian coordinates to spherical coordinates.

    Args:
        pts_cart (array_like): Cartesian :math:`x`, :math:`y` and
            :math:`z`. Of shape N-by-3 or length 3 if just one point.
        convention (str, optional): Convention for spherical coordinates:
            ``'lat-lng'`` or ``'theta-phi'``:

            .. code-block:: none

                   lat-lng
                                            ^ z (lat = 90)
                                            |
                                            |
                       (lng = -90) ---------+---------> y (lng = 90)
                                          ,'|
                                        ,'  |
                   (lat = 0, lng = 0) x     | (lat = -90)

            .. code-block:: none

                theta-phi
                                            ^ z (theta = 0)
                                            |
                                            |
                       (phi = 270) ---------+---------> y (phi = 90)
                                          ,'|
                                        ,'  |
                (theta = 90, phi = 0) x     | (theta = 180)

    Returns:
        numpy.ndarray: Spherical coordinates :math:`(r, \theta_1, \theta_2)`
        in radians.
    """
    pts_cart = np.array(pts_cart)

    # Validate inputs
    is_one_point = False
    if pts_cart.shape == (3,):
        is_one_point = True
        pts_cart = pts_cart.reshape(1, 3)
    elif pts_cart.ndim != 2 or pts_cart.shape[1] != 3:
        raise ValueError("Shape of input must be either (3,) or (n, 3)")

    # Compute r
    r = np.sqrt(np.sum(np.square(pts_cart), axis=1))

    # Compute latitude
    z = pts_cart[:, 2]
    lat = np.arcsin(z / r)

    # Compute longitude
    x = pts_cart[:, 0]
    y = pts_cart[:, 1]
    lng = np.arctan2(y, x) # choosing the quadrant correctly

    # Assemble
    pts_r_lat_lng = np.stack((r, lat, lng), axis=-1)

    # Select output convention
    if convention == 'lat-lng':
        pts_sph = pts_r_lat_lng
    elif convention == 'theta-phi':
        pts_sph = _convert_sph_conventions(
            pts_r_lat_lng, 'lat-lng_to_theta-phi')
    else:
        raise NotImplementedError(convention)

    if is_one_point:
        pts_sph = pts_sph.reshape(3)

    return pts_sph


def sph2cart(pts_sph, convention='lat-lng'):
    """Inverse of :func:`cart2sph`.

    See :func:`cart2sph`.
    """
    pts_sph = np.array(pts_sph)

    # Validate inputs
    is_one_point = False
    if pts_sph.shape == (3,):
        is_one_point = True
        pts_sph = pts_sph.reshape(1, 3)
    elif pts_sph.ndim != 2 or pts_sph.shape[1] != 3:
        raise ValueError("Shape of input must be either (3,) or (n, 3)")

    # Degrees?
    _warn_degree(pts_sph[:, 1:])

    # Convert to latitude-longitude convention, if necessary
    if convention == 'lat-lng':
        pts_r_lat_lng = pts_sph
    elif convention == 'theta-phi':
        pts_r_lat_lng = _convert_sph_conventions(
            pts_sph, 'theta-phi_to_lat-lng')
    else:
        raise NotImplementedError(convention)

    # Compute x, y and z
    r = pts_r_lat_lng[:, 0]
    lat = pts_r_lat_lng[:, 1]
    lng = pts_r_lat_lng[:, 2]
    z = r * np.sin(lat)
    x = r * np.cos(lat) * np.cos(lng)
    y = r * np.cos(lat) * np.sin(lng)

    # Assemble and return
    pts_cart = np.stack((x, y, z), axis=-1)

    if is_one_point:
        pts_cart = pts_cart.reshape(3)

    return pts_cart