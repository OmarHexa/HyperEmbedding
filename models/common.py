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

class CBS(nn.Module):
    def __init__(self,in_channel,out_channel,kernel=3,stride=1,group=1) -> None:
        super().__init__()
        if kernel==3:
            padd = 1
        elif kernel==5:
            padd =2
        else:
            padd = 0
        self.conv = nn.Conv2d(in_channel,out_channel,kernel_size=kernel,stride=stride,padding=padd,groups=group)
        self.bn = nn.BatchNorm2d(out_channel,eps=1e-03)
    def forward(self,x):
        x = self.conv(x)
        x= self.bn(x)
        x = F.silu(x)
        return x
    def fuseforward(self,x):
        return F.silu(self.conv(x))

class non_bottleneck_1d(nn.Module):
    def __init__(self, chann, dropprob=0.3, groups =2,dilation=1):
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(
            1*dilation, 0), bias=True,groups=groups, dilation=(dilation, 1))

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(
            0, 1*dilation), bias=True,groups=groups, dilation=(1, dilation))

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(
            2*(dilation), 0), bias=True, groups=groups,dilation=(2*dilation, 1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(
            0, 2*(dilation)), bias=True,groups=groups, dilation=(1, 2*dilation))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)
    def forward(self, input):

        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)
        return F.relu(output+input)  # +input = identity (residual connection)
    
class ELAN(nn.Module):
    def __init__(self, in_channel, out_channel,n=2) -> None:
        super().__init__()
        mid_channel = in_channel//2
        self.convleft = nn.Conv2d(in_channel, mid_channel, 1, 1)
        self.convright = nn.Conv2d(in_channel, mid_channel, 1, 1)
        self.comp1 = nn.Sequential(*[CBS(mid_channel,mid_channel,group=2) for _ in range(n)])
        self.comp2 = nn.Sequential(*[CBS(mid_channel,mid_channel,group=2) for _ in range(n)])
        self.agg = CBS(2*in_channel,out_channel,kernel=1)
    def forward(self, input):
        #channel partialization
        x_left = self.convleft(input).chunk(2,1)
        x_right =self.convright(input).chunk(2,1)
        #computational block
        x1 = self.comp1(x_right).chunk(2,1)
        x2 = self.comp2(x1).chunk(2,1)
        #channel shuffle
        xg1 = torch.cat((x_left[0],x_right[0],x1[0],x2[0]),dim=1)
        xg2 = torch.cat((x_left[1],x_right[1],x1[1],x2[1]),dim=1)  
        # aggregation
        x = self.agg(torch.cat((xg1,xg2),dim=1))
        return x
    
class ELAN_D(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()
        mid_channel = in_channel//2
        self.convleft = nn.Conv2d(in_channel, out_channel, 1, 1)
        self.convright = nn.Conv2d(in_channel, out_channel, 1, 1)
        self.conv1 = CBS(mid_channel,mid_channel//2,group=2)
        self.conv2 = CBS(mid_channel//2,mid_channel//2,group=2)
        self.conv3 = CBS(mid_channel//2,mid_channel//2,group=2)
        self.conv4 = CBS(mid_channel//2,mid_channel//2,group=2)
        self.agg = CBS(2*in_channel,out_channel,kernel=1)
    def forward(self, input):
        #channel partialization
        x_left = self.convleft(input)
        x_right =self.convright(input)
        #computational block
        x1 = self.conv1(x_right)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)
        x4 = self.conv4(x2)
        
        c_ = x_left.shape[1]//2
        #channel shuffle
        xg1 = torch.cat((x_left[:,:c_],x_right[:,:c_],x1[:,:c_],x2[:,:c_],x3[:,:c_],x4[:,:c_]),dim=1)
        xg2 = torch.cat((x_left[:,c_:],x_right[:,c_:],x1[:,c_:],x2[:,c_:],x3[:,c_:],x4[:,c_:]),dim=1)
        
        # aggregation
        x = self.agg(torch.cat((xg1,xg2),dim=1))
        return x

class SOCA(nn.Module):
    def __init__(self, channels):
        super().__init__()

        # Define the Covariance layer
        self.conv = nn.Conv1d(channels, channels, kernel_size=3,padding=1)
    # @timing
    def forward(self, x):
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
        attention = self.conv(cov)  # B x C x 1
        attention = torch.sigmoid(attention)  # B x C x 1

        # Apply the channel attention weights to the input tensor
        x_weighted = x.transpose(1, 2) * attention  # B x C x hw
        out = x_weighted.view(batch_size, channels,
                              height, width)  # B x C x H x W
        # print("SOCA",x.shape)
        return out
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


class DownsamplerBlock(nn.Module):
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
    
class SOECA(nn.Module):
    def __init__(self,channels,gamma=2,b=1) -> None:
        super().__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))    
        self.max_pool = nn.AdaptiveMaxPool2d((1,1))
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=int(k_size//2), bias=False)
        self.sigmoid = nn.Sigmoid()
    # @timing
    def forward(self, x):
        y1 = self.avg_pool(x).squeeze(-1)
        y2 = self.max_pool(x).squeeze(-1)
        cov = torch.matmul(y2, y1.transpose(1, 2))
        # Compute the channel-wise mean and covariance
        cov = torch.mean(cov, dim=1, keepdim=True)  # B x 1 X C
        # Compute the channel attention weights
        attention = self.conv(cov).transpose(-1, -2).unsqueeze(-1) 
        y = self.sigmoid(attention)
        return y
class ChannelSampler(nn.Module):
    def __init__(self,in_channel,out_channel=16) -> None:
        super().__init__()
        self.m = out_channel
        self.attn = SOECA(in_channel)
        self.norm = nn.InstanceNorm2d(out_channel*2)
        self.conv = CBS(out_channel*2,out_channel,3,2)
    def forward(self,x):
        score = torch.mean(self.attn(x),dim=0)
        score_id = torch.argsort(score,dim=0)
        max_id = score_id[-self.m:].squeeze()
        x1 = x[:, max_id]
        x2 = self._groupchannels(x,self.m)
        x = self.norm(torch.cat((x1,x2),dim=1))
        x = self.conv(x)
        return x
    def _groupchannels(self,input, m):
        b,c,h,w = input.shape
        g = c//m
        if c%m !=0:
            s = c-(g*m) 
            input = input[:,s:]
        output = input.view(b, m, g, h,w)
        output = torch.sum(output, axis=2)
        return output
    
class Upsampler(nn.Module):
    def __init__(self, in_channel, out_channel,mode ='Bilinear'):
        super().__init__()
        self.conv = CBS(in_channel, out_channel, 1)
        if mode == 'nearest':
            self.up = nn.UpsamplingNearest2d(scale_factor=2)
        elif mode == 'transpose':
            self.up = nn.ConvTranspose2d(out_channel, out_channel, 3)
        elif mode == 'Bilinear':
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
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = CBS(c1, 2 * self.c, 1)
        self.cv2 = CBS((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=3, e=0.5) for _ in range(n))
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=3, e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = CBS(c1, c_, k)
        self.cv2 = CBS(c_, c2, k)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    
class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = CBS(c1, c_, k, s, g)
        self.cv2 = CBS(c_, c_, 5, 1,c_)
    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1) 
    
class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = CBS(c1, c_, 1, 1)
        self.cv2 = CBS(c_ * 4, c2, 1, 1)
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
        self.conv = CBS(c1 * 4, c2, k, s, g)
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

