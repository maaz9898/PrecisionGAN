import torch
import torch.nn as nn
import math
import torch.nn.utils.spectral_norm as spectral_norm
from utils import get_norm_layer

class MultiHeadAttention(nn.Module):
    def __init__(self, in_dim, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_per_head = in_dim // num_heads

        assert self.dim_per_head * num_heads == in_dim, "in_dim must be divisible by num_heads"

        self.query_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        
        # Projection layer for residual connection
        self.proj_layer = nn.Conv2d(in_dim, in_dim, kernel_size=1)

        # Final linear transformation layer
        self.fc = nn.Conv2d(in_dim, in_dim, kernel_size=1)

    def transpose_for_scores(self, x):
        batch_size, channels, height, width = x.size()
        new_channels = channels // self.num_heads
        # New shape: [Batch, Num Heads, New Channels, Height, Width]
        new_x_shape = (batch_size, self.num_heads, new_channels, height, width)
        x = x.view(*new_x_shape)
        return x.permute(0, 1, 3, 4, 2)  # [Batch, Num Heads, Height, Width, New Channels]

    def forward(self, x):
        batch_size = x.size(0)

        # Applying convolutions
        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)

        # Splitting into multiple heads and transposing
        query = self.transpose_for_scores(query)
        key = self.transpose_for_scores(key)
        value = self.transpose_for_scores(value)

        # Scaled dot-product attention
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.dim_per_head)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # Apply attention to the value
        context = torch.matmul(attention_probs, value)

        # Combine heads
        context = context.permute(0, 1, 4, 2, 3).contiguous()
        new_context_shape = (batch_size, -1) + context.size()[3:]
        context = context.view(*new_context_shape)

        # Apply final linear transformation
        context = self.fc(context)

        # Project input x to the same dimension as the output of self.fc
        x_proj = self.proj_layer(x)

        # Add residual connection
        output = context + x_proj  # Element-wise addition
        return output

class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, nf=64, norm_layer=get_norm_layer(), use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            nf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(nf * 8, nf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        
        # add intermediate layers with ngf * 8 filters
        unet_block = UnetSkipConnectionBlock(nf * 8, nf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(nf * 8, nf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(nf * 8, nf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        
        # gradually reduce the number of filters from nf * 8 to nf
        unet_block = UnetSkipConnectionBlock(nf * 4, nf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(nf * 2, nf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(nf, nf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, nf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        output = self.model(input)
        # Check the output dimensions match the input dimensions
        if output.size(2) != input.size(2) or output.size(3) != input.size(3):
            raise ValueError(f"Generator output has incorrect spatial dimensions: {output.size()} expected {input.size()}.")
        return output
    
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, norm_layer=get_norm_layer(), use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
                
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        
        # Advanced Upsampling
        upconv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1, bias=False) 
        )

        # Attention Mechanism
        if not outermost and not innermost:
            self.attention = MultiHeadAttention(inner_nc * 2)
        else:
            self.attention = None
            
        # Model construction
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + [self.attention] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            x1 = self.model(x)
            # Concatenate the skip connection
            x_cat = torch.cat([x, x1], 1)
            return x_cat

# Define the Discriminator class with a dropout layer
class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=get_norm_layer(), dropout=0.5):
        super(Discriminator, self).__init__()
        sequence = [spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1)), nn.LeakyReLU(0.2, True)]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=False)),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
                nn.Dropout(dropout)  # Add dropout layer here
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1, bias=False)),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        
        sequence += [spectral_norm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)), nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

