import numpy as np
import cv2
import glob
import os,time
import torch.optim as optim
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from os import path
import pathlib
import torch.nn.functional as F
import einops
from einops import rearrange, repeat, reduce

from typing import Optional
import math
from Loss_vgg import SSIM
from tqdm import tqdm
from torch import einsum

print("Start")

class DS(Dataset):
    def __init__(self, tr_data):
        self.len = len(tr_data[0])
        self.x_data = tr_data[0]
        self.y_data = tr_data[1]

    def extract_patches(self, sub_short, sub_long):

        H = sub_short.shape[1]
        W = sub_short.shape[2]
        
        xx = np.random.randint(0,W-ps)
        yy = np.random.randint(0,H-ps)
        
        short_patch = sub_short[:,yy:yy+ps,xx:xx+ps]
        long_patch = sub_long[:,yy:yy+ps,xx:xx+ps]
        
        if np.random.randint(2,size=1)[0] == 1:  # random flip 
            short_patch = np.flip(short_patch, axis=1)
            long_patch = np.flip(long_patch, axis=1)
        if np.random.randint(2,size=1)[0] == 1: 
            short_patch = np.flip(short_patch, axis=0)
            long_patch = np.flip(long_patch, axis=0)
        if np.random.randint(2,size=1)[0] == 1:  # random transpose 
            short_patch = np.transpose(short_patch, (0,2,1))
            long_patch = np.transpose(long_patch, (0,2,1))
            
        short_patch = torch.tensor(short_patch.copy(), device=device, dtype=torch.float32)
        long_patch = torch.tensor(long_patch.copy(), device=device, dtype=torch.float32)
 
        return short_patch, long_patch

    def __getitem__(self, index):
        short_patch, long_patch = self.extract_patches(self.x_data[index], self.y_data[index])
        
        return short_patch, long_patch

    def __len__(self):
        return self.len

class SelfCrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None):
        super().__init__()
        self.dim_head = (int(dim / heads)) if dim_head is None else dim_head
        _dim = self.dim_head * heads
        self.heads = heads
        self.to_qvk = nn.Linear(dim, _dim * 3, bias=False)
        self.W_01 = nn.Linear( _dim, dim, bias=False)
        self.scale_factor = self.dim_head ** -0.5
        self.size=16
        self.tb= self.size**2 
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Sequential(
            nn.Linear(dim, 2*dim),
            nn.ReLU(),
            nn.Linear(2*dim, dim))        

    def forward(self, xp, xc, xf,xp_dilated,xf_dilated, height,width):

        xp_res= rearrange(xp, "n  (h w b1 b2) c -> n (h w) (b1 b2) c", b1=self.size,b2=self.size,h=height,w=width) 
        xc_res= rearrange(xc, "n  (h w b1 b2) c -> n (h w) (b1 b2) c", b1=self.size,b2=self.size,h=height,w=width) 
        xf_res= rearrange(xf, "n  (h w b1 b2) c -> n (h w) (b1 b2) c", b1=self.size,b2=self.size,h=height,w=width) 
        
        qkv_p = self.to_qvk(xp)  # [batch, tokens, dim*3*heads ]
        qkv_c = self.to_qvk(xc) 
        qkv_f = self.to_qvk(xf) 

        qkv_pdil = self.to_qvk(xp_dilated) 
        qkv_fdil = self.to_qvk(xf_dilated) 
        
        qp, kp, vp = tuple(rearrange(qkv_p, 'b (ht wt tb) (d k h) -> k b h (ht wt) tb d ', k=3,tb=self.tb,ht=height, h=self.heads))
        qc, kc, vc = tuple(rearrange(qkv_c, 'b (ht wt tb) (d k h) -> k b h (ht wt) tb d ', k=3,tb=self.tb,ht=height, h=self.heads))
        qf, kf, vf = tuple(rearrange(qkv_f, 'b (ht wt tb) (d k h) -> k b h (ht wt) tb d ', k=3,tb=self.tb,ht=height, h=self.heads))
        
        qp_dil, kp_dil, vp_dil = tuple(rearrange(qkv_pdil, 'b (ht wt tb) (d k h) -> k b h (ht wt) tb d ', k=3,tb=self.tb,ht=height, h=self.heads))
        qf_dil, kf_dil, vf_dil = tuple(rearrange(qkv_fdil, 'b (ht wt tb) (d k h) -> k b h (ht wt) tb d ', k=3,tb=self.tb,ht=height, h=self.heads))
        
        # Step 3
        # resulted shape will be: [batch, heads, tokens, tokens]
        sa_p  = torch.softmax(torch.einsum('b h n i d, b h n j d -> b h n i j', qp, kp) * self.scale_factor, dim=-1)
        sa_c  = torch.softmax(torch.einsum('b h n i d, b h n j d -> b h n i j', qc, kc) * self.scale_factor, dim=-1)
        sa_f  = torch.softmax(torch.einsum('b h n i d, b h n j d -> b h n i j', qf, kf) * self.scale_factor, dim=-1)

        ca_cp = torch.softmax(torch.einsum('b h n i d, b h n j d -> b h n i j', qc, kp) * self.scale_factor, dim=-1)
        ca_cf = torch.softmax(torch.einsum('b h n i d, b h n j d -> b h n i j', qc, kf) * self.scale_factor, dim=-1)

        ca_pc = torch.softmax(torch.einsum('b h n i d, b h n j d -> b h n i j', qp, kc) * self.scale_factor, dim=-1)
        ca_fc = torch.softmax(torch.einsum('b h n i d, b h n j d -> b h n i j', qf, kc) * self.scale_factor, dim=-1)

        ca_cp_dil = torch.softmax(torch.einsum('b h n i d, b h n j d -> b h n i j', qc, kp_dil) * self.scale_factor, dim=-1)
        ca_cf_dil = torch.softmax(torch.einsum('b h n i d, b h n j d -> b h n i j', qc, kf_dil) * self.scale_factor, dim=-1)



        
        # Step 4. Calc result per batch and per head h
        sa_p_out  = torch.einsum('b h n i j, b h n j d -> b h n i d', sa_p, vp)
        sa_c_out  = torch.einsum('b h n i j, b h n j d -> b h n i d', sa_c, vc)
        sa_f_out  = torch.einsum('b h n i j, b h n j d -> b h n i d', sa_f, vf)

        ca_cp_out = torch.einsum('b h n i j, b h n j d -> b h n i d', ca_cp, vp)
        ca_cf_out = torch.einsum('b h n i j, b h n j d -> b h n i d', ca_cf, vf)

        ca_pc_out = torch.einsum('b h n i j, b h n j d -> b h n i d', ca_pc, vc)
        ca_fc_out = torch.einsum('b h n i j, b h n j d -> b h n i d', ca_fc, vc)

        ca_cp_dil_out = torch.einsum('b h n i j, b h n j d -> b h n i d', ca_cp_dil, vp_dil)
        ca_cf_dil_out = torch.einsum('b h n i j, b h n j d -> b h n i d', ca_cf_dil, vf_dil)



        # Step 5. Re-compose: merge heads with dim_head d
        sa_p_out  = self.W_01(rearrange(sa_p_out, "b h nb tb d -> b nb tb (h d)"))
        sa_c_out  = self.W_01(rearrange(sa_c_out, "b h nb tb d -> b nb tb (h d)"))
        sa_f_out  = self.W_01(rearrange(sa_f_out, "b h nb tb d -> b nb tb (h d)"))

        ca_cp_out = self.W_01(rearrange(ca_cp_out, "b h nb tb d -> b nb tb (h d)"))
        ca_cf_out = self.W_01(rearrange(ca_cf_out, "b h nb tb d -> b nb tb (h d)"))
        ca_pc_out = self.W_01(rearrange(ca_pc_out, "b h nb tb d -> b nb tb (h d)"))
        ca_fc_out = self.W_01(rearrange(ca_fc_out, "b h nb tb d -> b nb tb (h d)"))

        ca_cp_dil_out = self.W_01(rearrange(ca_cp_dil_out, "b h nb tb d -> b nb tb (h d)"))
        ca_cf_dil_out = self.W_01(rearrange(ca_cf_dil_out, "b h nb tb d -> b nb tb (h d)"))


        sa_p_out=sa_p_out+ xp_res
        sa_c_out=sa_c_out+ xc_res
        sa_f_out=sa_f_out+ xf_res

        ca_cp_out=ca_cp_out+ xc_res
        ca_cf_out=ca_cf_out+ xc_res
        ca_pc_out=ca_pc_out+ xp_res
        ca_fc_out=ca_fc_out+ xf_res

        ca_cp_dil_out=ca_cp_dil_out+ xc_res
        ca_cf_dil_out=ca_cf_dil_out+ xc_res

        sa_p_out = self.norm(self.linear1(self.norm(sa_p_out))+ sa_p_out)
        sa_c_out = self.norm(self.linear1(self.norm(sa_c_out))+ sa_c_out)
        sa_f_out = self.norm(self.linear1(self.norm(sa_f_out))+ sa_f_out)

        ca_cp_out= self.norm(self.linear1(self.norm(ca_cp_out))+ ca_cp_out)
        ca_cf_out= self.norm(self.linear1(self.norm(ca_cf_out))+ ca_cf_out)

        ca_pc_out= self.norm(self.linear1(self.norm(ca_pc_out))+ ca_pc_out)
        ca_fc_out= self.norm(self.linear1(self.norm(ca_fc_out))+ ca_fc_out)

        ca_cp_dil_out= self.norm(self.linear1(self.norm(ca_cp_dil_out))+ ca_cp_dil_out)
        ca_cf_dil_out= self.norm(self.linear1(self.norm(ca_cf_dil_out))+ ca_cf_dil_out)




        
        # Step 6. Apply final linear transformation layer
        return sa_p_out,sa_c_out,sa_f_out,ca_cp_out,ca_cf_out, ca_pc_out, ca_fc_out,ca_cp_dil_out,ca_cf_dil_out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, dim_head=None):
        super(TransformerBlock, self).__init__()
        #dim_head = dim // num_heads
        dim_linear_block= dim*2
        self.mhsa = MultiHeadSelfAttention(dim,num_heads)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)

        self.linear = nn.Sequential(
           nn.Linear(dim, dim_linear_block),
           nn.ReLU(),
           nn.Linear(dim_linear_block, dim))

    def forward(self, x,h,w):
       
       y = self.norm_1(self.mhsa(x,h,w))
       return self.norm_2(self.linear(y) + y)


       #return  y

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None):
        super().__init__()
        self.dim_head = (int(dim / heads)) if dim_head is None else dim_head
        _dim = self.dim_head * heads
        self.heads = heads
        self.to_qvk = nn.Linear(dim, _dim * 3, bias=False)
        self.W_0 = nn.Linear( _dim, dim, bias=False)
        self.scale_factor = self.dim_head ** -0.5
        self.size=16
        self.tb= self.size**2 
    def forward(self, x, h,w,mask=None):
        assert x.dim() == 3

        x_res= rearrange(x, "n  (h w b1 b2) c -> n (h w) (b1 b2) c", b1=self.size,b2=self.size,h=h,w=w) 
        # Step 1
        qkv = self.to_qvk(x)  # [batch, tokens, dim*3*heads ]

        # Step 2
        # decomposition to q,v,k and cast to tuple
        # the resulted shape before casting to tuple will be:
        # [3, batch, heads, tokens, dim_head]
        q, k, v = tuple(rearrange(qkv, 'b t (d k h) -> k b h t d ', k=3, h=self.heads))

        q1 = rearrange(q, 'b h (ht wt tb) d -> b h (ht wt) tb d',tb=self.tb,ht=h)
        k1 = rearrange(k, 'b h (ht wt tb) d -> b h (ht wt) tb d',tb=self.tb,ht=h)
        v1 = rearrange(v, 'b h (ht wt tb) d -> b h (ht wt) tb d',tb=self.tb,ht=h)
        
        # Step 3
        # resulted shape will be: [batch, heads, tokens, tokens]
        #scaled_dot_prod = torch.einsum('b h nb tbi d, b h nb tbj d -> b h nb tbi tbj', q1, k1) * self.scale_factor
        scaled_dot_prod = torch.einsum('b h n i d, b h n j d -> b h n i j', q1, k1) * self.scale_factor

        # scaled_dot_prod = torch.einsum('b h i d , b h j d -> b h i j', q, k) * self.scale_factor

        attention = torch.softmax(scaled_dot_prod, dim=-1)
        

        # Step 4. Calc result per batch and per head h
        #out = torch.einsum('b h nb tbi tbj, b h nb tbj d -> b h nb tbi d', attention, v)
        out = torch.einsum('b h n i j, b h n j d -> b h n i d', attention, v1)


        # Step 5. Re-compose: merge heads with dim_head d
        out = rearrange(out, "b h nb tb d -> b nb tb (h d)")
        out_=self.W_0(out)
        

        # Step 6. Apply final linear transformation layer
        return out_+x_res

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def stable_softmax(t, dim = -1):
    t = t - t.amax(dim = dim, keepdim = True)
    return t.softmax(dim = dim)

def dilation(x):
    
    if x.shape[1]==256:
        size = 16

        x_hat= rearrange(x,  "n c (h b1) (w b2) -> n (h w) (b1 b2) c", b1=size,b2=size)
        
        x_cat= rearrange(x_hat,  "n (hw) (b1b2) c -> n (hw b1b2) c")
        

    else:

        size=32
        b= torch.nn.functional.pad(x, pad=(8, 8, 8, 8), mode='replicate')
        B1= rearrange(b[:,:,:-16,:-16],  "n c (h b1) (w b2) -> n (h w) (b1 b2) c", b1=size,b2=size)
        B2= rearrange(b[:,:,:-16,16:],   "n c (h b1) (w b2) -> n (h w) (b1 b2) c", b1=size,b2=size)
        B3= rearrange(b[:,:,16:,:-16],   "n c (h b1) (w b2) -> n (h w) (b1 b2) c", b1=size,b2=size)
        B4= rearrange(b[:,:,16:,16:],    "n c (h b1) (w b2) -> n (h w) (b1 b2) c", b1=size,b2=size)


        x_cat = torch.cat([B1,B2,B3,B4], axis= 2)
        x_cat= rearrange(x_cat,  "n (hw) (b1b2) c -> n (hw b1b2) c")

        x_cat = x_cat[:,::4,:]

    return x_cat

class fusion2(nn.Module):
    def __init__(self,out_channels):
        super().__init__()
        self.r=out_channels
        self.conv1= nn.Conv2d(out_channels,1, kernel_size= (1,1), stride= (1,1))
        self.conv2= nn.Conv2d(out_channels,1, kernel_size= (1,1), stride= (1,1))
        #self.conv3= nn.Conv2d(out_channels,1, kernel_size= (1,1), stride= (1,1))

    def forward(self, xc_sa,xc_cp):
        xc_sa_1= self.conv1(xc_sa)
        xc_cp_1= self.conv2(xc_cp)
        

        #------------Softmax-----------------------
        x_soft= torch.softmax(torch.cat([xc_sa_1,xc_cp_1],dim=1), dim=1)
        xc_sa_1,xc_cp_1= torch.chunk(x_soft,2,dim=1)

        #------------Repeat-----------------------
        xc_sa_r, xc_cp_r =map(lambda t: repeat(t, "n c h w  -> n (c r) h w", r= self.r), (xc_sa_1,xc_cp_1))

        #------------Convex Combination-----------------------
        x_cross =xc_sa * xc_sa_r + xc_cp * xc_cp_r 

        return x_cross

class fusion3(nn.Module):
    def __init__(self,out_channels):
        super().__init__()
        self.r=out_channels
        self.conv1= nn.Conv2d(out_channels,1, kernel_size= (1,1), stride= (1,1))
        self.conv2= nn.Conv2d(out_channels,1, kernel_size= (1,1), stride= (1,1))
        self.conv3= nn.Conv2d(out_channels,1, kernel_size= (1,1), stride= (1,1))

    def forward(self, xc_sa,xc_cp,xc_cf,cp_dilated,cf_dilated):
        xc_sa_1= self.conv1(xc_sa)
        xc_cp_1= self.conv2(xc_cp)
        xc_cf_1= self.conv3(xc_cf)
        xc_cp_dil_1= self.conv2(cp_dilated)
        xc_cf_dil_1= self.conv3(cf_dilated)

        #------------Softmax-----------------------
        x_soft= torch.softmax(torch.cat([xc_sa_1,xc_cp_1,xc_cf_1,xc_cp_dil_1,xc_cf_dil_1],dim=1), dim=1)
        xc_sa_1,xc_cp_1,xc_cf_1,xc_cp_dil_1,xc_cf_dil_1= torch.chunk(x_soft,5,dim=1)

        #------------Repeat-----------------------
        xc_sa_r, xc_cp_r, xc_cf_r,xc_cp_dil_r,xc_cf_dil_r =map(lambda t: repeat(t, "n c h w  -> n (c r) h w", r= self.r), (xc_sa_1,xc_cp_1,xc_cf_1,xc_cp_dil_1,xc_cf_dil_1))

        #------------Convex Combination-----------------------
        xc_cross =xc_sa * xc_sa_r + xc_cp * xc_cp_r + xc_cf * xc_cf_r + cp_dilated * xc_cp_dil_r + cf_dilated * xc_cf_dil_r

        return xc_cross

class ChannelAttention(nn.Module):
    def __init__(self, num_features, reduction=4):
        super(ChannelAttention, self).__init__()
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, num_features // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // reduction, num_features, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        
        return x * self.module(x)

class RCAB(nn.Module):
    def __init__(self, num_features):
        super(RCAB, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            ChannelAttention(num_features)
        )

    def forward(self, x):
        return x + self.module(x)

class MAB_Enc(nn.Module):
    def __init__(self,out_channels,num_heads):
        super().__init__()
        
        self.size= 16
        self.r=out_channels
        self.selfcross_attn= SelfCrossAttention(out_channels,num_heads)
        self.fusion2= fusion2(out_channels)
        self.fusion3= fusion3(out_channels)
        

    def forward(self, xp,xc,xf):
        
        height= int ((xp.shape[2])/self.size)
        width= int ((xp.shape[3])/self.size)
        xp_dilated= dilation(xp)
        xf_dilated= dilation(xf)
        xp_b,xc_b,xf_b =map(lambda t: rearrange(t, "n c (h b1) (w b2) -> n (h w b1 b2) c", b1=self.size,b2=self.size), (xp,xc,xf))

        xp_sa, xc_sa, xf_sa, xc_cp, xc_cf, xc_pc, xc_fc,cp_dil,cf_dil = self.selfcross_attn(xp_b, xc_b, xf_b,xp_dilated,xf_dilated,height,width)
    
        p_sa, c_sa, f_sa, cp, cf, pc,fc,cp_dilated,cf_dilated =map(lambda t: rearrange(t, "n (h w) (b1 b2) c-> n c (h b1) (w b2)", b1=self.size, h = height, w=width), (xp_sa, xc_sa, xf_sa, xc_cp, xc_cf,xc_pc, xc_fc,cp_dil,cf_dil))
        
        xp_cross= self.fusion2(p_sa,pc)
        xf_cross= self.fusion2(f_sa,fc)

        xc_cross= self.fusion3(c_sa,cp,cf,cp_dilated,cf_dilated)
       
        return xp_cross, xc_cross, xf_cross

class ENC(nn.Module):
    def __init__(self,in_channels, out_channels,num_heads):

        super().__init__()
        self.conv1= nn.Conv2d(in_channels= in_channels, out_channels= out_channels, kernel_size= (3,3), stride= (1,1), padding=(1,1))

        self.module = nn.Sequential(torch.nn.InstanceNorm2d(out_channels),
        nn.Conv2d(in_channels= out_channels, out_channels= out_channels, kernel_size= (1,1), stride= (1,1)),
        nn.GELU())
        self.MAB_blocks_Enc = nn.ModuleList([MAB_Enc(out_channels,num_heads) for i in range(2)])

        self.rcab = RCAB(out_channels)
        
   
    def forward(self, x):
        
        x0= self.conv1(x[0])
        residual0 = x0
        x0= self.module(x0)
        
        x1= self.conv1(x[1])
        residual1 = x1
        x1= self.module(x1)
        
        x2= self.conv1(x[2])
        residual2 = x2
        x2= self.module(x2)
        
        for MAB_block_Enc in self.MAB_blocks_Enc:
            res_mab_x0= x0
            res_mab_x1= x1
            res_mab_x2= x2
            x0,x1,x2 = MAB_block_Enc(x0,x1,x2)
            x0= x0+ res_mab_x0
            x1= x1+ res_mab_x1
            x2= x2+ res_mab_x2
            x0= self.rcab(x0)
            x1= self.rcab(x1)
            x2= self.rcab(x2)
        x0_res= x0+ residual0
        x1_res= x1+ residual1
        x2_res= x2+ residual2
        x=[x0_res,x1_res,x2_res]
        return x

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels,num_heads):
        super().__init__()
        
        self.maxpool= nn.MaxPool2d(2)
        self.ENC=ENC(in_channels,out_channels,num_heads)
    def forward(self, x):
        x0=self.maxpool(x[0])
        x1=self.maxpool(x[1])
        x2=self.maxpool(x[2]) 
        x= self.ENC([x0,x1,x2])
        #print("shape",x.shape)
        return x

class MAB_Dec(nn.Module):
    def __init__(self,out_channels,num_heads):
        super().__init__()
        self.size= 16
        self.transformer= TransformerBlock(out_channels,num_heads)
        
    def forward(self, x):
        
        height= int ((x.shape[2])/self.size)
        width= int ((x.shape[3])/self.size)
        
        x1_block = rearrange(x, "n c (h b1) (w b2) -> n (h w b1 b2) c", b1=self.size,b2=self.size)        
        x1_transformed = self.transformer(x1_block,height,width)
        x1_unblock = rearrange(x1_transformed, "n (h w) (b1 b2) c-> n c (h b1) (w b2)", b1=self.size,b2=self.size, h = height, w=width)

        return x1_unblock

class DEC(nn.Module):
    def __init__(self,in_channels, out_channels,num_heads):

        super().__init__()
        self.conv1= nn.Conv2d(in_channels= in_channels, out_channels= out_channels, kernel_size= (3,3), stride= (1,1), padding=(1,1))

        self.module = nn.Sequential(torch.nn.InstanceNorm2d(out_channels),
        nn.Conv2d(in_channels= out_channels, out_channels= out_channels, kernel_size= (1,1), stride= (1,1)),
        nn.GELU())
        self.MAB_blocks = nn.ModuleList([MAB_Dec(out_channels,num_heads) for i in range(2)])
       
        self.rcab= RCAB(out_channels)
        
   
    def forward(self, x):
        x= self.conv1(x)
        residual1 = x
        x= self.module(x)
        
        for MAB_block in self.MAB_blocks:
        
            res_mab= x
            x = MAB_block(x)
            x= x+ res_mab
            x = self.rcab(x)
        x= x + residual1
        return x

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels,num_heads, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DEC(in_channels, out_channels,num_heads)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DEC(in_channels, out_channels,num_heads)

    def forward(self, x1, x2):
        
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Multiscale_ENC(nn.Module):
    def __init__(self):

        super().__init__()

        self.ENC  = ENC(3,32,2)
        self.Down2= Down(32,64,4)
        self.Down3= Down(64,128,8)
        self.Down4= Down(128,256,16)
        self.Bottelneck  = ENC(256,256,16) 
        self.up1 = Up(512,128,16)
        self.up2 = Up(256,64,8)
        self.up3 = Up(128,32,4)
        self.up4 = Up(64,32,2)
        
        self.conv3x3= nn.Conv2d(in_channels= 32, out_channels= 3, kernel_size= (3,3), stride= (1,1), padding=(1,1))
        #self.norm = nn.LayerNorm(norm_len)
        
   
    def forward(self, x):
        
        x1= self.ENC(x)
        x2= self.Down2(x1)
        x3= self.Down3(x2)
        x4= self.Down4(x3)
        #x5= self.Down5(x4)
        x5= self.Bottelneck(x4)
        
        x = self.up1(x5[1], x4[1])
        x = self.up2(x, x3[1])
        x = self.up3(x, x2[1])
        x = self.up4(x, x1[1])
        x= self.conv3x3(x)
        
        return x  
        
def load_fn_example(train_id,ps,height,width):

    #put the directory according to your Dataset path 

    data_dir_ip = '/Dataset/DRV_RGB_short/'
    data_dir_gt = '/Dataset/long_short/'

    
    i1 = cv2.imread(f'{data_dir_ip}{train_id}/0.png', cv2.IMREAD_UNCHANGED)
    i2 = cv2.imread(f'{data_dir_ip}{train_id}/1.png', cv2.IMREAD_UNCHANGED)
    i3 = cv2.imread(f'{data_dir_ip}{train_id}/2.png', cv2.IMREAD_UNCHANGED)
    
    in_im = np.concatenate((i1, i2, i3), axis= 2)

    in_im= np.moveaxis(in_im, (0,1,2), (1,2,0))
    in_im= in_im/255
    
    
    gt = cv2.imread(f'{data_dir_gt}{train_id}/0.png', cv2.IMREAD_UNCHANGED)
    #gt = cv2.imread(gt_files[0], cv2.IMREAD_UNCHANGED)
   
    gt_im = np.moveaxis(gt, (0,1,2), (1,2,0))
    gt_im= gt_im/255

    return in_im, gt_im

lr = 1e-4
batch_size=2
num_epochs = 901
save_freq = 100
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
train_ids = [line.rstrip('\n') for line in open('train_list.txt')]
ps= 384
height=832
width= 1248
num= int((height/ps)*(width/ps))

n=len(train_ids)
input_data = np.zeros([n , 3*3, height, width])
gt_data = np.zeros([n, 3, height, width])



save_dir = 'checkpoints'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


#
print("data loading...")
for i in range(n):
    ip,gt = load_fn_example(train_ids[i],ps,height,width)
    input_data[i:(i+1),:,:,:] = ip
    gt_data[i:(i+1),:,:,:] = gt

model = Multiscale_ENC().to(device=device)
model= nn.DataParallel(model,device_ids=[1,2])

overall_loss = SSIM().to(device=device)

model.train()

use_amp = True
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler= MultiStepLR(optimizer, milestones=[1000], gamma=0.1)
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)



Data = [input_data, gt_data]
print('len', (Data[0].shape))
dataset = DS(Data)
tr_lod = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

avg_loss = torch.zeros(num_epochs)
print('training started')
for epoch in range(num_epochs):
    total_loss = 0
    for i, (input_data, target) in tqdm(enumerate(tr_lod)):
        optimizer.zero_grad()
        input_data = input_data.to(device=device)
        target = target.to(device=device)
        ip0=input_data[:,:3]
        ip1=input_data[:,3:6]
        ip2=input_data[:,6:9]
        ip= [ip0,ip1,ip2]
        
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            output = model(ip)
            loss= overall_loss(target,output)

        total_loss += loss.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.nn.utils.clip_grad_value_(model.parameters(), 1.)

    scheduler.step()
        
    avg_loss[epoch] = total_loss / len(tr_lod)
    print(f'{epoch}/{num_epochs}==>VGG loss: {avg_loss[epoch]:03}')

    if epoch%save_freq==0:
        torch.save(model, os.path.join(save_dir, 'DRV_%03d.pth' % (epoch+1)))
torch.save(model.state_dict(), f'{save_dir}/DRV_{num_epochs}_batch_size_{batch_size}_.pth')
print('weight_saved')
avg_loss = avg_loss.detach().numpy()
print('Training Completed')



