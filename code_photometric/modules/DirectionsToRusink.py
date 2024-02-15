import torch
import torch.nn.functional as F


def DirectionsToRusink(light, view=None, normal=None, t=None, output_ch=2):
    if view is None:
        view = torch.zeros_like(light)
        view[..., 2] = 1
        view = view.detach()
    light, view, normal = F.normalize(light, p=2, dim=-1), F.normalize(view, p=2, dim=-1), F.normalize(normal, p=2, dim=-1)
    H = F.normalize((view + light) / 2, p=2, dim=-1)
    theta_h = (normal * H).sum(dim=-1) - 0.5
    theta_d = (view * H).sum(dim=-1) - 0.5
    return torch.stack([theta_h, theta_d][:output_ch], dim=-1)

