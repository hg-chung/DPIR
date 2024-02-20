import torch
from modules.embedder import *
import numpy as np
import torch.nn as nn


class SpecularNetwork(torch.nn.Module):
    def __init__(
            self,
            d_in, # input dimension (nv,vh): 2 
            dims,
            skip_in=(),
            weight_norm=False,
            multires=0, 
            d_out=9 # num basis: n
    ):
        super().__init__()
        
        dims = [d_in] + dims + [(d_out) * 3]
        
        self.embed_fn = None
        self.d_out = d_out 
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires,1)
            self.embed_fn = embed_fn
            dims[0] = input_ch
            
        self.num_layers = len(dims)
        self.skip_in = skip_in
        
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)
            
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            
            setattr(self, "lin" + str(l), lin)
        
        self.relu = torch.nn.functional.relu

    def forward(self, input):
        if self.embed_fn is not None:
            input = self.embed_fn(input)

        x = input
        
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)
        outputs = torch.clamp(torch.abs(x),0,3) 

        outputs = outputs.view(outputs.size(0), self.d_out , -1)   # (batch, num_basis, channel)
        return outputs