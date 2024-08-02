import torch
from torch import nn
from torch.nn import functional as F
import collections

from . import ResBlock2D, AFNOBlock2d


class AFNOConditionerNetBase(nn.Module):
    def __init__(
        self,
        autoencoder,
        embed_dim=128,
        embed_dim_out=None,
        analysis_depth=4,
        forecast_depth=4,
        input_size_ratios=(1,),
        train_autoenc=False,
        afno_fusion=False,
    ):
        super().__init__()
        
        self.train_autoenc = train_autoenc
        if not isinstance(autoencoder, collections.abc.Sequence):
            autoencoder = [autoencoder]
        num_inputs = len(autoencoder)
        if not isinstance(embed_dim, collections.abc.Sequence):
            embed_dim = [embed_dim] * num_inputs
        if embed_dim_out is None:
            embed_dim_out = embed_dim[0]
        if not isinstance(analysis_depth, collections.abc.Sequence):
            analysis_depth = [analysis_depth] * num_inputs
        self.embed_dim = embed_dim
        self.embed_dim_out = embed_dim_out

        # encoding + analysis for each input
        self.autoencoder = nn.ModuleList()
        self.proj = nn.ModuleList()
        self.analysis = nn.ModuleList()

        for i in range(num_inputs):
            ae = autoencoder[i].requires_grad_(train_autoenc)
            self.autoencoder.append(ae)

            proj = nn.Conv2d(ae.encoded_channels//2, embed_dim[i], kernel_size=1)
            self.proj.append(proj)

            analysis = nn.Sequential(
                *(AFNOBlock2d(embed_dim[i]) for _ in range(analysis_depth[i]))
            )
            self.analysis.append(analysis)

        # data fusion
        self.fusion = FusionBlock2d(embed_dim, input_size_ratios,
            afno_fusion=afno_fusion, dim_out=embed_dim_out)


    def forward(self, x):
        (x, t_relative) = list(zip(*x))

        # encoding + analysis for each input
        def process_input(i):
            z = self.autoencoder[i].encode(x[i])[0]
            z = self.proj[i](z)
            z = z.permute(0,2,3,1)
            z = self.analysis[i](z)
            return z

        x = [process_input(i) for i in range(len(x))]
        
        if len(x) > 1:
            # merge inputs
            x = self.fusion(x)
        else:
            x = x[0]

        return x.permute(0,3,1,2) # to channels-first order


class AFNOConditionerNetCascade(AFNOConditionerNetBase):
    def __init__(self, *args, cascade_depth=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.cascade_depth = cascade_depth
        self.resnet = nn.ModuleList()        
        ch = self.embed_dim_out
        self.cascade_dims = [ch]
        for i in range(cascade_depth-1):
            ch_out = 2*ch
            self.cascade_dims.append(ch_out)
            self.resnet.append(
                ResBlock2D(ch, ch_out, kernel_size=(3,3), norm=None)
            )
            ch = ch_out

    def forward(self, x):
        x = super().forward(x)
        # print(f"post AFNONowcastNetBase forward: {x.shape}")
        img_shape = tuple(x.shape[-2:])
        cascade = {img_shape: x}
        for i in range(self.cascade_depth-1):
            x = F.avg_pool2d(x, (2,2))
            x = self.resnet[i](x)
            img_shape = tuple(x.shape[-2:])
            cascade[img_shape] = x
        return cascade


class FusionBlock2d(nn.Module):
    def __init__(self, dim, size_ratios, dim_out=None, afno_fusion=False):
        super().__init__()

        N_sources = len(size_ratios)
        if not isinstance(dim, collections.abc.Sequence):
            dim = (dim,) * N_sources
        if dim_out is None:
            dim_out = dim[0]
        
        self.scale = nn.ModuleList()
        for (i,size_ratio) in enumerate(size_ratios):
            if size_ratio == 1:
                scale = nn.Identity()
            else:
                scale = []
                while size_ratio > 1:
                    scale.append(nn.ConvTranspose2d(
                        dim[i], dim_out if size_ratio==2 else dim[i],
                        kernel_size=(3,3), stride=(2,2),
                        padding=(1,1), output_padding=(1,1)
                    ))
                    size_ratio //= 2
                scale = nn.Sequential(*scale)
            self.scale.append(scale)

        self.afno_fusion = afno_fusion
        
        if self.afno_fusion:
            if N_sources > 1:
                self.fusion = nn.Sequential(
                    nn.Linear(sum(dim), sum(dim)),
                    # AFNOBlock2d(dim*N_sources, mlp_ratio=2),
                    AFNOBlock2d(sum(dim), mlp_ratio=2),
                    nn.Linear(sum(dim), dim_out)
                )
            else:
                self.fusion = nn.Identity()
        
    def resize_proj(self, x, i):
        x = x.permute(0,3,1,2)
        x = self.scale[i](x)
        x = x.permute(0,2,3,1)
        # print(f"post resize_proj: {x.shape}")
        return x

    def forward(self, x):
        x = [self.resize_proj(xx, i) for (i, xx) in enumerate(x)]
        if self.afno_fusion:        
            x = torch.concat(x, axis=-1)
            x = self.fusion(x)
            # print(f"post afno_fusion: {x.shape}")
        else:
            x = sum(x)
            # x = torch.cat(x, axis = -1)    
        return x


