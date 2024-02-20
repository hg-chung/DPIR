

import torch
import torch.nn.functional as F
import numpy as np
from pytorch3d.ops import knn_points
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor
)
from pyhocon import ConfigFactory

from modules.geometry_network import GeometryNetwork
from modules.reflectance_network import DiffuseNetwork, CoeffNetwork
from modules.specular_network import SpecularNetwork
from modules.DirectionsToRusink import DirectionsToRusink

from modules.utils import device, load_mem_data, load_test_data, \
    get_rays, remove_outlier


class CoreModel(torch.nn.Module):
    def __init__(self, args):
        super(CoreModel, self).__init__()
        self.args = args
        self.raster_n = args.raster_n 
        self.dataname = args.dataname 
        conf = ConfigFactory.parse_file(args.conf)
        self.conf = conf.get_config('model')
        self.splatting_r = self.conf.get_float('splatting_r')
        self.data_r = self.conf.get_float('data_r')
        pointcloud, imagesgt, K, R, T1, T2, poses, masks = load_mem_data(args.memitem)
        self.img_s = imagesgt.shape[1] 
        self.R, self.T1, self.T2, self.K = R, T1, T2, K 
        self.imagesgt = imagesgt 
        self.masks = masks  
        self.H, self.W = self.imagesgt.shape[1],self.imagesgt.shape[2]
        
        N = int(pointcloud.shape[0] * self.data_r)
        ids = np.random.permutation(pointcloud.shape[0])[:N] 
        pointcloud = pointcloud[ids][:, :3] # initiallized point number
        print('Initialized point number:{}'.format(pointcloud.shape[0]))
        
        self.geometry_network = GeometryNetwork(**self.conf.get_config('geometry_network'))
        self.diffuse_network = DiffuseNetwork(**self.conf.get_config('reflectance_network'))
        self.coeff_network = CoeffNetwork(**self.conf.get_config('reflectance_network'))
        self.specular_network = SpecularNetwork(**self.conf.get_config('specular_network'))
        self.vertsparam = torch.nn.Parameter(torch.Tensor(pointcloud[:, :3]), ) 
        self.point_radius = torch.nn.Parameter(torch.Tensor(torch.ones((self.vertsparam.shape[0]))*self.splatting_r),requires_grad=True)
        self.viewdir = []
        self.raster_settings = PointsRasterizationSettings( 
            bin_size=30, # size of bins for coarse to fine rasterization
            image_size=self.img_s, # img size
            radius=self.point_radius, # radius 
            points_per_pixel=self.raster_n, # number of points to keep track of per pixel
        )
        self.compositor = AlphaCompositor().cuda() 
        self.onlybase = False

    def repeat_pts(self): 
        self.vertsparam.data = self.vertsparam.data.repeat(2,1) 
        self.point_radius.data = self.point_radius.data.repeat(2)
        
        self.raster_settings = PointsRasterizationSettings(  
            bin_size=30, 
            image_size=self.img_s, 
            radius=self.point_radius, 
            points_per_pixel=self.raster_n, 
        )
        
        if self.vertsparam.grad is not None:
            self.vertsparam.grad = self.vertsparam.grad.repeat(2,1) 
        if self.point_radius.grad is not None:
            self.point_radius.grad = self.point_radius.grad.repeat(2)
        self.point_radius = torch.nn.Parameter(self.point_radius)
        print("number of upsampled points",self.vertsparam.data.shape)
        print("radius shape",self.point_radius.data.shape)

    def remove_out(self): # 
        pts_all = self.vertsparam.data 
        pts_in = remove_outlier(pts_all.cpu().data.numpy()) 
        pts_in = torch.tensor(pts_in).cuda().float() 
        idx = knn_points(pts_in[None,...], pts_all[None,...], None, None, 1).idx[0,:,0] 
        
        self.vertsparam.data = self.vertsparam.data[idx].detach() 
        self.point_radius.data = self.point_radius.data[idx].detach()
        print("number of downsampled points",self.vertsparam.data.shape)
        print("radius shape",self.point_radius.data.shape)
        
        if self.vertsparam.grad is not None:
            self.vertsparam.grad = self.vertsparam.grad[idx].detach()
        if self.point_radius.grad is not None:
            self.point_radius.grad = self.point_radius.grad[idx].detach()  
    
    def compute_normals_and_feature_vectors(self):
        p = self.vertsparam.detach() 
        eikonal_points = p
        eikonal_output, grad_thetas = self.geometry_network.gradient(eikonal_points.detach())
        n_points = self.vertsparam.shape[0] 
        normals = torch.nn.functional.normalize(grad_thetas[:n_points, :], dim=1) # normals
        geometry_output = self.geometry_network(self.vertsparam.detach())  
        sdf_values = geometry_output[:, 0] 
        feature_vectors = geometry_output[:,1:]

        self._output['sdf_values'] = sdf_values
        self._output['feature_vectors'] = feature_vectors
    
        return normals,feature_vectors # normals and feature vectors 
    
    def _render(self, point_cloud, cameras): 
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=self.raster_settings)
        fragments = rasterizer(point_cloud) # point cloud -> rasterizer -> fragments
        dists2 = fragments.dists.permute(0, 3, 1, 2) 
        indices = fragments.idx.long().permute(0, 3, 1, 2) 
        mask = indices !=-1 # mask for not rendered points
        frag_r = torch.zeros_like(dists2)
        frag_r[mask] = self.point_radius[indices[mask]]

        alphas = torch.zeros_like(dists2)
        alphas[mask] = 1 - dists2[mask] / (frag_r[mask] * frag_r[mask]) # alphas with mask
        images = self.compositor( #images, weight (N,C,H,W)
            fragments.idx.long().permute(0, 3, 1, 2), # (N,H,W,points_per_pixel) --> (N,points_per_pixel,H,W)
            alphas,
            point_cloud.features_packed().permute(1, 0), # (C,N)
        )
        images = images.permute(0, 2, 3, 1) # images (N,H,W,C)
        return images
        
        
    def forward(self, id, vis_train =False):
        self._output={}
        cam_loc = torch.tensor(self.T1[id:id+1])
        cameras = PerspectiveCameras(focal_length=((float(self.K[0][0]/self.H*2),float(self.K[1][1]/self.W*2)),),
                                 principal_point = (((self.H/2-float(self.K[0][2]))/self.H*2, -(self.W/2-float(self.K[1][2]))/self.W*2),),
                                    device=device, R=-self.R[id:id + 1], T=-self.T2[id:id + 1]) 
        normals, feature_vectors = self.compute_normals_and_feature_vectors()
        view_dir = cam_loc - self.vertsparam.detach()
        view_dir = view_dir.reshape(-1,3)
        view_dir = view_dir / (torch.norm(view_dir, dim=-1, keepdim=True) + 1e-6)
        light_dir = view_dir         
        vis = torch.ones_like(self.vertsparam[...,[0]]) # mask
        halfangle = DirectionsToRusink(light= light_dir, view=view_dir, normal=normals, output_ch = 1)
        albedo = self.diffuse_network(self.vertsparam.detach(),feature_vectors)
        spec_coeff, coeff_norm = self.coeff_network(self.vertsparam.detach(),feature_vectors)
        self._output['spec_coeff']= coeff_norm
        spec_basis = self.specular_network(halfangle)
        spec_ref = torch.einsum("ijk,ij->ik",spec_basis,spec_coeff)
        brdf = albedo + spec_ref
        render_shading = F.relu((normals * light_dir).sum(dim=-1, keepdims=True)) # cos
        features = torch.clamp(render_shading * brdf*0.2, 0., 1.) 
        albedo_features = torch.clamp(albedo*0.2,0.,1.)
        spec_features = torch.clamp(spec_ref*render_shading*0.2,0.,1.)
        
        features = torch.cat([features, normals, albedo_features, spec_features, vis], dim=-1)
    
        point_cloud = Pointclouds(points=[self.vertsparam], features=[features]) # 3d points with points and features
        feat = self._render(point_cloud, cameras).flip(1)
        
        base, normal_img, albedo, spec_ref, vis = feat[..., :3], feat[..., 3:6], feat[..., 6:9], feat[..., 9:12], feat[..., 12:]
        image = base
        normal_img = normal_img.squeeze()
        normal_img  = torch.einsum('ij,hwj->hwi',torch.linalg.inv(torch.tensor(self.R[id])),normal_img)
        normal_img = normal_img.reshape(1,self.H,self.W,3)
        return image.clamp(min=0, max=1), normal_img, albedo, spec_ref, vis, self._output

