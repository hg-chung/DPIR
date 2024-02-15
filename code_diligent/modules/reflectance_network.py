import torch
from modules.embedder import *
import numpy as np
import torch.nn as nn


class DiffuseNetwork(nn.Module):
    def __init__(
            self,
            d_in,
            feature_vector_size,
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0,
            coeff_reg = 1.0
    ):
        super().__init__()

        dims = [d_in] + dims
        self.feature_vector_size = feature_vector_size
        self.embed_fn = None
        self.multires = multires
        self.bias = bias
        
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires,3)
            self.embed_fn = embed_fn
            dims[0] = input_ch + feature_vector_size
            
        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - input_ch
            else:
                out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)
             
        self.relu = nn.functional.relu

        self.fc_diff = nn.Linear(256, 3)
        
    def forward(self, input, feature_vectors =None):
        if self.embed_fn is not None:
            input = self.embed_fn(input)
        x = torch.cat([input,feature_vectors],dim=-1)

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 1:
                x = self.relu(x)
                

        albedo = torch.abs(self.fc_diff(x)*10)
        return albedo
    
class CoeffNetwork(nn.Module):
    def __init__(
            self,
            d_in,
            feature_vector_size,
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0,
            num_basis =9,
            coeff_reg = 0.2
    ):
        super().__init__()

        dims = [d_in] + dims
        self.feature_vector_size = feature_vector_size
        self.embed_fn = None
        self.multires = multires
        self.bias = bias
        self.coeff_reg = coeff_reg
        
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires,3)
            self.embed_fn = embed_fn
            dims[0] = input_ch + feature_vector_size
            
        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - input_ch
            else:
                out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)
             
        self.relu = nn.functional.relu
        
        self.fc_spec_coeff = nn.Linear(256, num_basis)

        
    def forward(self, input, feature_vectors =None):
        if self.embed_fn is not None:
            input = self.embed_fn(input)
        x = torch.cat([input,feature_vectors],dim=-1)

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 1:
                x = self.relu(x)
                
        spec_coeff = torch.clamp(torch.abs(self.fc_spec_coeff(x)),0.,self.coeff_reg)
        coeff_norm = (torch.abs(spec_coeff.norm(p=1,dim=-1)-self.coeff_reg)).mean()
        return spec_coeff, coeff_norm