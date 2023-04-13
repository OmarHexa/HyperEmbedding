import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import wraps
from time import time
from torch.profiler import profile, record_function, ProfilerActivity
from matplotlib import pyplot as plt
from models.common import *
# from common import *



# def timing(f):
#     @wraps(f)
#     def wrap(*args, **kw):
#         ts = time()
#         result = f(*args, **kw)
#         te = time()
#         print(f'Function {f.__name__} took {te-ts:2.4f} seconds')
#         return result
#     return wrap
def run_model(model, data):
    outputs = model(data)
# taken from pytorch : https://discuss.pytorch.org/t/gpu-memory-that-model-uses/56822
def ModelSize(model):
    param_size = sum([param.nelement()*param.element_size() for param in model.parameters()])
    buffer_size = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('Model size: {:.3f}MB'.format(size_all_mb))

class EncoderBlock(nn.Module):
    def __init__(self, in_channel,out_channel,n=3) -> None:
        super().__init__()
        self.down = DownsamplerBlock(in_channel, out_channel)
        self.compblock = C2f(out_channel, out_channel,n=n,shortcut=True)
        self.attention = CBAM(out_channel)
    def forward(self, x):
        x = self.down(x)
        x = self.attention(x)
        x = self.compblock(x)
        return x
class EncoderBlock2(nn.Module):
    def __init__(self, in_channel,out_channel,n=3) -> None:
        super().__init__()
        self.down = DownsamplerBlock(in_channel, out_channel)
        self.compblock = C2FA(out_channel, out_channel,n=n,shortcut=True)
    def forward(self, x):
        x = self.down(x)
        x = self.compblock(x)
        return x
class DecoderBlock2(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()
        self.comp = C2FA(in_channel,in_channel)
        self.up = Upsampler(in_channel, out_channel)
    def forward(self, x,x_skip):
        x = self.comp(torch.sum((x,x_skip)))
        x = self.up(x)
        return x
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()
        self.skip_conv = CBS(in_channel,in_channel)
        self.comp = C2f(2*in_channel,in_channel)
        self.up = Upsampler(in_channel, out_channel)
    def forward(self, x,x_skip):
        x_skip = self.skip_conv(x_skip)
        x = self.comp(torch.cat((x,x_skip),dim=1))
        x = self.up(x)
        return x


class HyperEncoder(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.stem = ChannelSampler(in_channel,32)
        self.encoder = nn.ModuleList()
        self.encoder.append(EncoderBlock(32,64,n=3))
        self.encoder.append(EncoderBlock(64,128,n=6))
        self.encoder.append(EncoderBlock(128,256,n=6))
        self.encoder.append(SPPF(256,256))

    def forward(self, input):
        en1 = self.stem(input)
        en2 = self.encoder[0](en1)
        en3 = self.encoder[1](en2)
        en4 = self.encoder[2](en3)
        en4 = self.encoder[3](en4)
        return  en4,en3,en2
class HyperEncoder2(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.stem = Focus(in_channel,16)
        self.encoder = nn.ModuleList()
        self.encoder.append(EncoderBlock2(16,64,n=3))
        self.encoder.append(EncoderBlock2(64,128,n=6))
        self.encoder.append(EncoderBlock2(128,256,n=6))
        self.encoder.append(SPPF(256,256))
    # @timing
    def forward(self, input):
        en1 = self.stem(input)
        en2 = self.encoder[0](en1)
        en3 = self.encoder[1](en2)
        en4 = self.encoder[2](en3)
        en4 = self.encoder[3](en4)
        # print("Encoder")
        return  en4,en3,en2
class HyperDecoder2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.up = Upsampler(256,128)
        self.decode1 = DecoderBlock(128,64)
        self.decode2 = DecoderBlock(64,32)
        self.output_conv = nn.ConvTranspose2d(
            32, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)
    # @timing
    def forward(self, en4,en3,en2):
        output= self.up(en4)
        output = self.decode1(output,en3)
        output = self.decode2(output,en2)
        output = self.output_conv(output)
        # print("Decoder")
        return output
class HyperDecoder (nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.up = Upsampler(256,128)
        self.decode1 = DecoderBlock(128,64)
        self.decode2 = DecoderBlock(64,32)
        self.output_conv = nn.ConvTranspose2d(
            32, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)
    def forward(self, en4,en3,en2):
        output= self.up(en4)
        output = self.decode1(output,en3)
        output = self.decode2(output,en2)
        output = self.output_conv(output)
        return output
    
class HyperNet(nn.Module):
    def __init__(self, in_channel,num_classes):  # use encoder to pass pretrained encoder
        super().__init__()

        self.encoder = HyperEncoder(in_channel)
        self.decoder = HyperDecoder(num_classes)
    def forward(self, input, only_encode=False):
        en3,en2,en1 = self.encoder(input)  # predict=False by default
        return self.decoder(en3,en2,en1)




if __name__ == "__main__":
    input = torch.randn(20, 154, 416, 416)
    # input = torch.randn(20, 64, 104, 104)
    
    model = HyperNet(154,5)
    # model = CBAM(154)
    model.eval()
    output= model(input)
    # # ModelSize(model)
    print(output.shape)
    
  