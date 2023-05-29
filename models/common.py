import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from functools import wraps
from time import time
from torch.profiler import profile, record_function, ProfilerActivity
from matplotlib import pyplot as plt
import math
def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f'Function {f.__name__} took {te-ts:2.4f} seconds')
        return result
    return wrap

# class CBS(nn.Module):
#     def __init__(self,in_channel,out_channel,kernel=3,stride=1,group=1) -> None:
#         super().__init__()
#         if kernel==3:
#             padd = 1
#         elif kernel==5:
#             padd =2
#         elif kernel==7:
#             padd=3
#         else:
#             padd = 0
#         self.conv = nn.Conv2d(in_channel,out_channel,kernel_size=kernel,stride=stride,padding=padd,groups=group)
#         self.bn = nn.BatchNorm2d(out_channel,eps=1e-03)
#     # @timing
#     def forward(self,x):
#         x = self.conv(x)
#         x= self.bn(x)
#         x = F.silu(x)
#         return x
#     def fuseforward(self,x):
#         return F.silu(self.conv(x))
class CBA(nn.Module):
    def __init__(self,in_channel,out_channel,kernel=3,stride=1,group=1,activation=nn.SiLU()) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channel,out_channel,kernel_size=kernel,stride=stride,padding=(kernel-1)//2,groups=group)
        self.bn = nn.BatchNorm2d(out_channel,eps=1e-03)
        self.act = activation
    def forward(self,x):
        return self.act(self.bn(self.conv(x)))
    def fuseforward(self,x):
        return self.act(self.conv(x))
    

# class SOCA(nn.Module):
#     def __init__(self, channels):
#         super().__init__()

#         # Define the Covariance layer
#         self.conv = nn.Conv1d(channels, channels, kernel_size=3,padding=1)
#     # @timing
#     def forward(self, x):
#         # Reshape the input tensor
#         batch_size, channels, height, width = x.shape
#         # B x C x hw ----> B x hw x C
#         x = x.view(batch_size, channels, height*width).transpose(1, 2)

#         # Compute the channel-wise mean and covariance
#         avg = torch.mean(x, dim=1, keepdim=True)  # B x 1 X C
#         x_centered = x - avg  # B x hw x C
#         cov = torch.matmul(x_centered.transpose(
#             1, 2), x_centered) / (height*width - 1)  # B x C x C

#         cov = torch.mean(cov, dim=2, keepdim=True)  # B x C X 1
#         # Compute the channel attention weights
#         attention = self.conv(cov)  # B x C x 1
#         attention = torch.sigmoid(attention)  # B x C x 1

#         # Apply the channel attention weights to the input tensor
#         x_weighted = x.transpose(1, 2) * attention  # B x C x hw
#         out = x_weighted.view(batch_size, channels,
#                               height, width)  # B x C x H x W
#         # print("SOCA",x.shape)
#         return out
class LCA(nn.Module):
    # Efficient Channel attention (Local)
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))    
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=int(k_size//2), bias=False)
        self.sigmoid = nn.Sigmoid()
    # @timing
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)
    
class LSA(nn.Module):
    # Local Spatial-attention module
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()
        
    def forward(self, x):
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))

class GCNet(nn.Module):
    def __init__(self,in_channel,reduction_ratio=8) -> None:
        super().__init__()
        mid_channel = in_channel//reduction_ratio
        self.theta = nn.Conv2d(in_channel,out_channels=1,kernel_size=1)
        self.reduction = nn.Conv2d(in_channel,mid_channel,1,1)
        self.ln = nn.LayerNorm([mid_channel,1,1])
        self.restore = nn.Conv2d(mid_channel,in_channel,1,1)
    # @timing
    def forward(self,input):
        b, c, h, w = input.size()
        theta = self.theta(input).view(b,1,h*w,1).permute(0,2,1,3) #Hw x 1 x 1
        theta = nn.functional.softmax(theta, dim=1) #Hw x 1 x 1
        phi = input.view(b,c,h*w) # c x HW
        alpha = torch.einsum('bch,bhij->bcij',phi,theta)
        alpha = self.reduction(alpha)
        alpha = F.relu(self.ln(alpha))
        alpha = self.restore(alpha)
        
        output = input + alpha
        
        return output


class Downsampler(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel-in_channel,
                              (3, 3), stride=2, padding=1, bias=True)
        
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(out_channel, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], dim=1)
        output = self.bn(output)
        return F.silu(output)
    


# class ECA(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.channel = channels
#         self.avg_pool = nn.AdaptiveAvgPool2d((1,1))    
#         self.max_pool = nn.AdaptiveMaxPool2d((1,1))
#         self.l1 = nn.Linear(channels, channels)
#         self.sigmoid = nn.Sigmoid()
#     def forward(self, x):
#         y1 = self.avg_pool(x).view(-1,self.channel)
#         y2 = self.max_pool(x).view(-1,self.channel)
#         y1_att = self.l1(y1)
#         y2_att = self.l1(y2)
#         return self.sigmoid(y1_att+y2_att).view(-1,self.channel,1,1)

class SOCA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((20,20))
        self.linear = nn.Linear(channels, channels)
    def forward(self, x):
        x = self.avg(x)
        # Reshape the input tensor
        batch_size, channels, height, width = x.shape
        # B x C x hw ----> B x hw x C
        x = x.view(batch_size, channels, height*width).transpose(1, 2)
        # Compute the channel-wise mean and covariance
        avg = torch.mean(x, dim=1, keepdim=True)  # B x 1 X C
        x_centered = x - avg  # B x hw x C
        cov = torch.matmul(x_centered.transpose(
            1, 2), x_centered) / (height*width - 1)  # B x C x C
        cov = torch.mean(cov, dim=2, keepdim=True)  # B x C X 1
        # Compute the channel attention weights
        return torch.sigmoid(self.linear(cov.squeeze(-1))).view(-1,channels,1,1)
        
# class BSA(nn.Module):
#     def __init__(self,in_channel,out_channel=32) -> None:
#         super().__init__()
#         self.mid = out_channel
#         self.attn = SOCA(in_channel)
#         self.conv = CBA(out_channel,out_channel,3,2)
#         self.align = STN(out_channel)
#     def forward(self,x):
#         score = torch.mean(self.attn(x),dim=0)
#         score_id = torch.argsort(score, dim=0, descending=True).squeeze()
#         max_id,_ = torch.sort(score_id[:self.mid],dim=0)
#         # max_score, max_id = torch.topk(score, k=self.m, dim=0)
#         x1 = x[:,max_id]*score[max_id]
#         # x2 = self._groupchannels(x,self.mid)
#         # x = self.conv(self.align(torch.cat((x1,x2),dim=1)))
#         return self.conv(self.align(x1))
    
#     def _groupchannels(self,input, m):
#         b,c,h,w = input.shape
#         g = c//m
#         if c%m !=0:
#             s = c-(g*m) 
#             input = input[:,s:]
#         output = input.view(b, m, g, h,w)
#         output = torch.mean(output, dim=2)
#         return output
class BSA(nn.Module):
    def __init__(self,in_channel,out_channel=32) -> None:
        super().__init__()
        self.mid = out_channel//2
        self.chattn = SOCA(in_channel)
        self.conv = CBA(out_channel,out_channel,3,2)
        self.attn = LSA()
        print("channel attention has groupings")
    def forward(self,x):
        score = torch.mean(self.chattn(x),dim=0)
        score_id = torch.argsort(score, dim=0, descending=True).squeeze()
        max_id,_ = torch.sort(score_id[:self.mid],dim=0)
        x1 = x[:,max_id]*(1+score[max_id])
        x2 = self._groupchannels(x,self.mid)
        x = torch.cat((x1,x2),dim=1)
        return self.conv(self.attn(x))
    
    def _groupchannels(self,input, m):
        b,c,h,w = input.shape
        g = c//m
        if c%m !=0:
            s = c-(g*m) 
            input = input[:,s:]
        output = input.view(b, m, g, h,w)
        output = torch.mean(output, dim=2)
        return output
    
# class Upsampler(nn.Module):
#     def __init__(self, ninput, noutput):
#         super().__init__()
#         self.conv = nn.ConvTranspose2d(
#             ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
#         self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

#     def forward(self, input):
#         output = self.conv(input)
#         output = self.bn(output)
#         return F.relu(output)

class Upsampler(nn.Module):
    def __init__(self, in_channel, out_channel,mode ='bilinear'):
        super().__init__()
        self.conv = CBA(in_channel, out_channel, 1)
        if mode == 'nearest':
            self.up = nn.UpsamplingNearest2d(scale_factor=2)
        elif mode == 'transpose':
            self.up = nn.ConvTranspose2d(out_channel, out_channel, 3)
        elif mode == 'bilinear':
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.up(x)
        return x
    
class CBAM(nn.Module):
    # Convolutional Block Attention Module
    def __init__(self, c1, kernel_size=7):  # ch_in, kernels
        super().__init__()
        self.channel_attention = LCA(c1)
        self.spatial_attention = LSA(kernel_size)
    def forward(self, x):
        return self.spatial_attention(self.channel_attention(x))

class GCA(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 1x1 convolution to generate query and key
        self.query_conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3,padding=1)
        self.key_conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3,padding=1)
        
        # sigmoid activation function
        self.sigmoid = nn.Sigmoid()
        
        # softmax activation function
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        # reshape to hwxc
        b, c, h, w = x.size()
        x_reshape = x.view(b, c, -1).permute(0, 2, 1)
        x = self.global_pool(x).squeeze(-1).permute(0,2,1)
        # global average pooling to have 1xc dimension
        query = self.query_conv(x)
        key = self.key_conv(x)
        
        # apply sigmoid on both query and key
        query = self.sigmoid(query)
        key = self.sigmoid(key)
        
        # multiply query and key to generate cxc attention map
        attn_map = torch.bmm(query.permute(0,2,1), key)
        
        # apply softmax on attention map
        attn_map = self.softmax(attn_map)
        
        # multiply attention map with input
        x_attn = torch.bmm(x_reshape, attn_map).permute(0, 2, 1).view(b, c, h, w)
        
        return x_attn

class GSA(nn.Module):
    def __init__(self, in_channels, reduced_channels=16):
        super().__init__()
        
        # Generate value, query and key
        self.value_conv = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.query_conv = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        
        # 1x1 convolution to get back to original channel shape
        self.conv_out = nn.Conv2d(reduced_channels, in_channels, kernel_size=1)
        
    def forward(self, x):
        batch_size, _, height, width = x.size()
                
        # Generate value, query and key
        value = self.value_conv(x).view(batch_size, -1, height*width)
        query = self.query_conv(x).view(batch_size, -1, height*width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, height*width)
        
        # Matrix multiply query and key and apply softmax to generate the attention map
        attention_map = torch.bmm(query, key)
        attention_map = torch.softmax(attention_map, dim=-1)
        
        # Multiply the attention map with value
        attended_values = torch.bmm(value, attention_map.permute(0, 2, 1)).view(batch_size, -1, height, width)
        
        # Do one more 1x1 convolution to get back to original channel shape
        output = self.conv_out(attended_values)
        
        return output

class GLAM(nn.Module):
    def __init__(self,in_channel) -> None:
        super().__init__()
        self.Lc = LCA(in_channel) 
        self.Ls = LSA(in_channel) #1xhxw
        self.Gc = GCA(in_channel) #cxhxw (applied on the input)
        self.Gs = GSA(in_channel)
        self.weights = nn.Parameter(torch.ones(3))
    # @timing
    def forward(self,input):
        
        Al_c = self.Lc(input) #cx1x1
        Al_s = self.Ls(input) #1xhxw
        G_c = self.Gc(input) #cxhxw
        G_s = self.Gs(input) #cxhxw
        Fl = input* Al_c + input
        Fl = Fl*Al_s + Fl
        Fg = input*G_c
        Fg = Fg*G_s + Fg
        output = self.weights[0]* Fl + self.weights[1]*Fg+self.weights[2]* input
        return output
        
class C2f(nn.Module):
    # CSP Bottleneck with 2 convolutions
    """This c2f is used in best rgb model """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = CBA(c1, 2 * self.c, 1)
        self.cv2 = CBA((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=3, e=0.5) for _ in range(n))
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    
# class C2FA(nn.Module):
#     # CSP Bottleneck with 2 convolutions
#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=1):  # ch_in, ch_out, number, shortcut, groups, expansion
#         super().__init__()
#         self.c = int(c2 * 0.5)  # hidden channels
#         self.cv1 = CBA(c1, 2 * self.c, 1)
#         self.cv2 = CBA((2 + n) * self.c, c2, 1)
#         self.sa = LSA(7)
#         self.ca = LCA((2 + n) * self.c)
#         self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, k=3,g=g, e=e) for _ in range(n))
#     # @timing   
#     def forward(self, x):
#         y = self.sa(self.cv1(x))
#         y = list(y.chunk(2,1))
#         y.extend(m(y[-1]) for m in self.m)
#         y = self.ca(torch.cat(y, 1))
#         return self.cv2(y)
# class C2f(nn.Module):
#     """CSP Bottleneck with 2 convolutions."""
#     def __init__(self, c1, c2, n=3, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
#         super().__init__()
#         self.c = int(c2 * e)  # hidden channels
#         self.cv1 = CBA(c1, 2 * self.c, 1, 1)
#         self.cv2 = CBA((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
#         self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, k=3,g=g, e=1.0) for _ in range(n))
#     def forward(self, x):
#         """Forward pass through C2f layer."""
#         y = list(self.cv1(x).chunk(2, 1))
#         y.extend(m(y[-1]) for m in self.m)
#         return self.cv2(torch.cat(y, 1))
    
class C2fSG(C2f):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=3, shortcut=False, g=4, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1,c2,n,shortcut,g,e)
        self.m = nn.ModuleList(BottleneckSG(c1, c1, shortcut, k=3,g=g, e=e) for _ in range(n))


class C2FA(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=1):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.cv1 = CBA(c1, 2 * c1, 1)
        self.cv2 = CBA((2 + n) * c1, c2, 1)
        self.sa = LSA(7)
        self.ca = LCA((2 + n) * c1)
        self.m = nn.ModuleList(Bottleneck(c1, c1, shortcut, k=3,g=g, e=e) for _ in range(n))
    # @timing   
    def forward(self, x):
        y = self.sa(self.cv1(x))
        y = list(y.chunk(2,1))
        y.extend(m(y[-1]) for m in self.m)
        y = self.ca(torch.cat(y, 1))
        return self.cv2(y)
class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=3, e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = CBA(c1, c_, k)
        self.cv2 = CBA(c_, c2, k)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
class BottleneckSG(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, k=3, g=4, e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = CBA(c1, c_, k,group=1)
        self.shuffle = ShuffleBlock(g)
        self.cv2 = CBA(c_, c2, k,group=g)
        self.add = shortcut and c1 == c2
    def forward(self, x):
        return x + self.cv2(self.shuffle(self.cv1(x))) if self.add else self.cv2(self.shuffle(self.cv1(x)))
class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups
    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N,C,H,W = x.size()
        g = self.groups
        return x.view(N,g,C//g,H,W).permute(0,2,1,3,4).reshape(N,C,H,W)        
class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = CBA(c1, c_, k, s, g)
        self.cv2 = CBA(c_, c_, 5, 1,c_)
    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1) 
    
class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = CBA(c1, c_, 1, 1)
        self.cv2 = CBA(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))
class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=3, s=1, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = CBA(c1 * 4, c2, k, s, g)
    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        y = torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1)
        return self.conv(y)
    
class DFocus(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates=[1, 2, 3]):
        super(DFocus, self).__init__()
        # assert out_channels % 3 == 0, "Output channels must be divisible by 3"
        self.out_channels = out_channels
        self.r1_channels = out_channels // 3
        self.r2_channels = out_channels // 3
        self.r3_channels = out_channels - self.r1_channels - self.r2_channels
        
        self.conv_r1 = nn.Conv2d(in_channels, self.r1_channels, kernel_size=3, stride=2, padding=dilation_rates[0]*1, dilation=dilation_rates[0])
        self.conv_r2 = nn.Conv2d(in_channels, self.r2_channels, kernel_size=3, stride=2, padding=dilation_rates[1]*1, dilation=dilation_rates[1])
        self.conv_r3 = nn.Conv2d(in_channels, self.r3_channels, kernel_size=3, stride=2, padding=dilation_rates[2]*1, dilation=dilation_rates[2])
        
    def forward(self, x):
        out_r1 = self.conv_r1(x)
        out_r2 = self.conv_r2(x)
        out_r3 = self.conv_r3(x)
        out = torch.cat([out_r1, out_r2, out_r3], dim=1)
        return out

class STN(nn.Module):
    def __init__(self,channel):
        super().__init__()
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(channel, 8, kernel_size=7),
            nn.MaxPool2d(2,stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.AdaptiveMaxPool2d((7,7)),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 7 * 7, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 7 * 7)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(),align_corners=True)
        return F.grid_sample(x, grid,align_corners=True)
class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = CBA(2, 1, kernel_size)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out) 
        return x * scale
class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial=no_spatial
        if not no_spatial:
            self.hw = AttentionGate()
    def forward(self, x):
        x_perm1 = x.permute(0,2,1,3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0,2,1,3).contiguous()
        x_perm2 = x.permute(0,3,2,1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0,3,2,1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1/3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1/2 * (x_out11 + x_out21)
        return x_out