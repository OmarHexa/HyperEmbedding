"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import torch.nn as nn
import models.erfnet as erfnet
import models.hypernet as hypernet
from torch.autograd import Function


class BranchedERFNet(nn.Module):
    def __init__(self, in_channel,num_classes, encoder=None):
        super().__init__()

        print('Creating branched erfnet with {} classes'.format(num_classes))

        if (encoder is None):
            self.encoder = erfnet.Encoder(in_channel,sum(num_classes))
        else:
            self.encoder = encoder

        self.decoders = nn.ModuleList()
        for n in num_classes:
            self.decoders.append(erfnet.Decoder(n))

    def init_output(self, n_sigma=1):
        with torch.no_grad():
            output_conv = self.decoders[0].output_conv
            print('initialize last layer with size: ',
                  output_conv.weight.size())

            output_conv.weight[:, 0:2, :, :].fill_(0)
            output_conv.bias[0:2].fill_(0)

            output_conv.weight[:, 2:2+n_sigma, :, :].fill_(0)
            output_conv.bias[2:2+n_sigma].fill_(1)

    def forward(self, input, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            output = self.encoder(input)

        return torch.cat([decoder.forward(output) for decoder in self.decoders], 1)


class BranchedHyperNet(nn.Module):
    def __init__(self, in_channel,num_classes, encoder=None):
        super().__init__()

        print('Creating branched hypernet with {} classes'.format(num_classes))

        if (encoder is None):
            self.encoder = hypernet.HyperEncoder(in_channel)
            
        else:
            self.encoder = encoder

        self.decoders = nn.ModuleList()
        for n in num_classes:
            self.decoders.append(hypernet.HyperDecoder(n))

    def init_output(self, n_sigma=1):
        with torch.no_grad():
            output_conv = self.decoders[0].output_conv
            print('initialize last layer with size: ',
                  output_conv.weight.size())

            output_conv.weight[:, 0:2, :, :].fill_(0)
            output_conv.bias[0:2].fill_(0)

            output_conv.weight[:, 2:2+n_sigma, :, :].fill_(0)
            output_conv.bias[2:2+n_sigma].fill_(1)

    def forward(self, input, only_encode=False):
        if only_encode:
            return self.encoder.forward(input)
        else:
            features = self.encoder(input)

        return torch.cat([decoder.forward(*features) for decoder in self.decoders], 1), features


class Discriminator(nn.Module):
    def __init__(self,channel=256) -> None:
        super().__init__()
        self.reversal= GradientReversal()
        self.pool= nn.AdaptiveAvgPool2d((10, 10))
        self.disc =nn.Sequential(
                        nn.Linear(channel * 10 * 10, 100),
                        nn.BatchNorm1d(100),
                        nn.ReLU(True),
                        nn.Linear(100,2),
                        nn.LogSoftmax(dim=-1))
        
    def forward(self,x,alpha):
        c = x.size(1)
        x= self.pool(self.reversal(x,lambda_=alpha))
        return self.disc(x.view(-1,c*10*10))
        
# https://www.kaggle.com/code/balraj98/unsupervised-domain-adaptation-by-backpropagation
class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(nn.Module):
    def __init__(self):
        super(GradientReversal, self).__init__()

    def forward(self, x,lambda_=1):
        return GradientReversalFunction.apply(x,lambda_)