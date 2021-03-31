#!/usr/bin/env python
# coding: utf-8

# In[6]:


import torch
from torch import nn,einsum
from einops import rearrange, repeat,reduce
from einops.layers.torch import Rearrange
from torch.nn import functional as F
from linformer import Linformer
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from torch import optim
torch.cuda.device(0)


# In[7]:


class Residual(nn.Module):
    def __init__(self,fn):
        super(Residual,self).__init__()
        self.fn=fn
    def forward(self,x,**kwargs):
        return self.fn(x,**kwargs)+x
class PreNorm(nn.Module):
    def __init__(self,dim,fn):
        super(PreNorm,self).__init__()
        self.norm=nn.LayerNorm(dim)
        self.fn=fn
    def forward(self,x,**kwargs):
        return self.fn(self.norm(x),**kwargs)
class FeedForward(nn.Module):
    def __init__(self,dim,hidden_dim,dropout=0.):
        super(FeedForward,self).__init__()
        self.net=nn.Sequential(
            nn.Linear(dim,hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,dim),
            nn.Dropout(dropout)
        )
    def forward(self,x):
        return self.net(x)
    
class Attention(nn.Module):
    def __init__(self,dim,heads=8,dim_head=64,dropout=0.):
        super(Attention,self).__init__()
        inner_dim=dim_head*heads
        project_out=not(heads==1 and dim_head==dim)
        
        self.heads=heads
        self.scale=dim_head**-0.5
        self.to_qkv=nn.Linear(dim,inner_dim*3,bias=False)
        self.to_out=nn.Sequential(
            nn.Linear(inner_dim,dim),
            nn.Dropout(dropout)
        )if project_out else nn.Identity()
    def forward(self,x,mask=None):
        b,n,_,h=*x.shape,self.heads
        qkv=self.to_qkv(x).chunk(3,dim=-1)
        q,k,v=map(lambda t:rearrange(t,'b n (h d) -> b h n d',h=h),qkv)
        
        dots=einsum('b h i d, b h j d -> b h i j',q,k)*self.scale
        mask_value=-torch.finfo(dots.dtype).max
        
        if mask is not None:
            mask=F.pad(mask,flatten(1),(1,0),value=True)
            assert mask.shape[-1]==dots.shape[-1],'mask has incorrect dimensions'
            mask=rearrange(mask,'b,i->b() i()')*rearrange(mask,'b j->b() ()j')
            dots.masked_fill_(~mask,mask_value)
            del mask
        attn=dots.softmax(dim=-1)
        out=einsum('b h i j, b h j d -> b h i d', attn, v)
        out=rearrange(out,'b h n d->b n (h d)')
        out=self.to_out(out)
        return out
class Transformer(nn.Module):
    def __init__(self,dim,depth,heads,dim_head,mlp_dim,dropout=0.):
        super(Transformer,self).__init__()
        self.layers=nn.ModuleList([])
        self.depth=depth
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim,Attention(dim,heads=heads,dim_head=dim_head,dropout=dropout))),
                Residual(PreNorm(dim,FeedForward(dim,mlp_dim,dropout=dropout)))
            ]))
    def forward(self,x,mask=None):
        for attn,ff in self.layers:
            x=attn(x,mask=mask)
            x=ff(x)
        return x

class PreData(nn.Module):
    def __init__(self,image_size,patch_size,channels=3,dim=128,dim_head=64,emb_dropout=0.):
        super(PreData,self).__init__()
        assert image_size%patch_size==0,'Image dimensions must be divisible by the patch size.'
        num_patches=(image_size//patch_size)**2
        patch_dim=channels*patch_size**2
        
        self.to_patch_embedding=nn.Sequential(
            Rearrange('b c (h p1)(w p2)->b (h w)(p1 p2 c)',p1=patch_size,p2=patch_size),
            nn.Linear(patch_dim,dim)
        )
        self.pos_embedding=nn.Parameter(torch.randn(1,num_patches,dim))
        self.dropout=nn.Dropout(emb_dropout)
    def forward(self,img,mask=None):
        x=self.to_patch_embedding(img)
        b,n,_=x.shape
        x+=self.pos_embedding[:,:n]
        x=self.dropout(x)
        return x


# In[8]:


class Convs(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Convs,self).__init__()
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=1,bias=True)
        self.bn=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=1,bias=True)
        self.bn=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(inplace=True)
        self.up=nn.Upsample(scale_factor=2,mode='bilinear')
    def forward(self,x):
        x=self.up(self.conv2(self.bn(self.conv1(x))))
        return x


# In[9]:


class Restore(nn.Module):
    def __init__(self,image_size,in_channels,out_channels1):
        super(Restore,self).__init__()
        self.origin_size=image_size
        self.image_size=image_size
        self.up1=Convs(in_channels,out_channels1)
        self.up2=Convs(out_channels1,out_channels1)
    def forward(self,x):  #x []
        b,c,w,h=x.shape
        x=self.up1(x)
#         print("x.shape",x.shape,self.image_size,4*w)
        while self.image_size!=8*w:
            self.image_size//=2
            x=self.up2(x)
#             print(w,self.image_size,x.shape)
        self.image_size=self.origin_size
        return x
    
class Residual2(nn.Module):
    def __init__(self,fn,dim,last_channel):
        super(Residual2,self).__init__()
        self.linear=nn.Linear(dim,last_channel)
        self.fn=fn
    def forwawrd(self,z,x):
        print(z.shape,x.shape)
        x=self.fn(self.linear(x))
        print(z.shape,x.shape)
        return z+x


# In[10]:


class SERT(nn.Module):
    def __init__(self,num_classes,image_size,patch_size,last_channel,dim_head,dim,depth,heads,mlp_dim,out_channel1,out_channel2):
        super(SERT,self).__init__()
        self.num_patches=image_size//patch_size
        self.predata=PreData(image_size,patch_size)
        self.transformer=Transformer(dim,depth,heads,dim_head,mlp_dim,dropout=0.)
        self.multiply=nn.Sequential(
            nn.Conv2d(out_channel1,out_channel1,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channel1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel1,out_channel2,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channel2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4,mode="bilinear")
        )
        self.to_latent=nn.Identity()
        self.mlp_head=nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim,last_channel)
       )
        self.linear=nn.Linear(dim,last_channel)
        self.reshape=nn.Sequential(
            Rearrange('b (np1 np2) l->b l np1 np2',np1=self.num_patches),
            Restore(image_size,last_channel,out_channel1)
        )
        self.conv=nn.Conv2d(out_channel2*4,num_classes,kernel_size=1)
#         self.residual2=Residual2(reshape,dim,last_channel)

    def forward(self,x): 
        x1=self.transformer(self.predata(x))
        x2=self.transformer(x1)
        x3=self.transformer(x2)
        x4=self.transformer(x3)
        #print(x1.shape,x2.shape,x3.shape,x4.shape)
        x1=self.reshape(self.mlp_head(self.to_latent(x1)))
        x2=self.reshape(self.mlp_head(self.to_latent(x2)))
        x3=self.reshape(self.mlp_head(self.to_latent(x3)))
        x4=self.reshape(self.mlp_head(self.to_latent(x4)))
        #print(x1.shape,x2.shape,x3.shape,x4.shape)
        z1=self.multiply(x1)
        z2=self.multiply(x1+x2)
        z3=self.multiply(x2+x3)
        z4=self.multiply(x3+x4)
        #print(z1.shape,z2.shape,z3.shape,z4.shape)
        
        z=self.conv(torch.cat((z1,z2,z3,z4),1))
        #print(z.shape)
        return z

