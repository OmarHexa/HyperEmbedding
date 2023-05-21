"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import torch.nn as nn
import models.erfnet as erfnet
import models.hypernet as hypernet
import models.multimodalnet as multimodalnet



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
            print('initialize last layer with size: git s',
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
            self.encoder = hypernet.HyperEncoder2(in_channel)
            
        else:
            self.encoder = encoder

        self.decoders = nn.ModuleList()
        for n in num_classes:
            self.decoders.append(hypernet.HyperDecoder2(n))

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

        return torch.cat([decoder.forward(*output) for decoder in self.decoders], 1)



class BranchedMultiModalNet(nn.Module):
    def __init__(self, in_channel,num_classes, encoder=None):
        super().__init__()

        print('Creating branched hypernet with {} classes'.format(num_classes))

        if (encoder is None):
            self.encoder = multimodalnet.HyperEncoder(in_channel)
        else:
            self.encoder = encoder

        self.decoders = nn.ModuleList()
        for n in num_classes:
            self.decoders.append(multimodalnet.HyperDecoder(n))

    def init_output(self, n_sigma=1):
        with torch.no_grad():
            output_conv = self.decoders[0].output_conv
            print('initialize last layer with size: ',
                  output_conv.weight.size())

            output_conv.weight[:, 0:2, :, :].fill_(0)
            output_conv.bias[0:2].fill_(0)

            output_conv.weight[:, 2:2+n_sigma, :, :].fill_(0)
            output_conv.bias[2:2+n_sigma].fill_(1)
    def forward(self, hs, rgb):
        features = self.encoder(hs,rgb)

        return torch.cat([decoder.forward(*features) for decoder in self.decoders], 1), features

