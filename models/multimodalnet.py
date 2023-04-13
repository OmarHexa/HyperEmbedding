import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from models.common import *



class EncoderBlock(nn.Module):
    def __init__(self, in_channel,out_channel,n=3) -> None:
        super().__init__()
        self.down = DownsamplerBlock(in_channel, out_channel)
        self.compblock = C2f(out_channel, out_channel,n=n,shortcut=True)
        self.stn = STN(out_channel)
    def forward(self, x):
        x = self.down(x)
        x = self.compblock(x)
        x - self.stn(x)
        return x

class EncoderBlock2(nn.Module):
    def __init__(self, in_channel,out_channel,n=3) -> None:
        super().__init__()
        self.down = DownsamplerBlock(in_channel, out_channel)
        self.compblock = C2f(out_channel, out_channel,n=n,shortcut=True)
    def forward(self, x):
        x = self.down(x)
        x = self.compblock(x)
        return x
class DecoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()
        self.comp = C2f(in_channel,in_channel)
        self.up = Upsampler(in_channel, out_channel)
    def forward(self, x,x_skip):
        x = self.comp(x+x_skip)
        x = self.up(x)
        return x

class HSEncoder(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.stem = ChannelSampler(in_channel,32)
        self.encoder = nn.ModuleList()
        self.encoder.append(EncoderBlock(32,64,n=3))
        self.encoder.append(EncoderBlock(64,128,n=6))
        self.encoder.append(EncoderBlock(128,256,n=6))
    def forward(self, input):
        en1 = self.stem(input)
        en2 = self.encoder[0](en1)
        en3 = self.encoder[1](en2)
        en4 = self.encoder[2](en3)
        return  en2,en3,en4

class RGBEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = Focus(3,32)
        self.encoder = nn.ModuleList()
        self.encoder.append(EncoderBlock2(32,64,n=3))
        self.encoder.append(EncoderBlock2(64,128,n=6))
        self.encoder.append(EncoderBlock2(128,256,n=6))
    def forward(self, input):
        en1 = self.stem(input)
        en2 = self.encoder[0](en1)
        en3 = self.encoder[1](en2)
        en4 = self.encoder[2](en3)
        return  en2,en3,en4
class HyperDecoder (nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.up = Upsampler(256,128)
        self.decode1 = DecoderBlock(128,64)
        self.decode2 = DecoderBlock(64,32)
        self.output_conv = nn.ConvTranspose2d(
            32, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)
    def forward(self, en2,en3,en4):
        output= self.up(en4)
        output = self.decode1(output,en3)
        output = self.decode2(output,en2)
        output = self.output_conv(output)
        return output
class HyperEncoder (nn.Module):
    def __init__(self, hs_channel):
        super().__init__()
        self.hs = HSEncoder(hs_channel)
        self.rgb = RGBEncoder()
        self.fusion1 = CMAFF(64)
        self.fusion2 = CMAFF(128)
        self.fusion3 = CMAFF(256)
        self.attn = GLAM(256)
    def forward(self, hs,rgb):
        hs1,hs2,hs3 = self.hs(hs)
        rgb1,rgb2,rgb3 = self.rgb(rgb)
        en2 = self.fusion1(hs1,rgb1)
        en3 = self.fusion2(hs2,rgb2)
        en4 = self.fusion3(hs3,rgb3)
        en4 = self.attn(en4)
        return en2,en3,en4
