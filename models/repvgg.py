# https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/modules/repvgg_block.py

from typing import Type, Union, Mapping, Any, Optional

import numpy as np
from matplotlib import pyplot as plt

import torch
from torch import nn
from models.common import *
# from common import *

class RepVGGBlock(nn.Module):
    """
    Repvgg block consists of three branches
    3x3: a branch of a 3x3 Convolution + BatchNorm + Activation
    1x1: a branch of a 1x1 Convolution + BatchNorm + Activation
    no_conv_branch: a branch with only BatchNorm which will only be used if
        input channel == output channel and use_residual_connection is True
    (usually in all but the first block of each stage)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        activation_type: Type[nn.Module] = nn.ReLU,
        activation_kwargs: Union[Mapping[str, Any], None] = None,
        se_type: Type[nn.Module] = nn.Identity,
        se_kwargs: Union[Mapping[str, Any], None] = None,
        build_residual_branches: bool = True,
        use_residual_connection: bool = True,
        use_alpha: bool = False,
    ):
        """

        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param activation_type: Type of the nonlinearity
        :param se_type: Type of the se block (Use nn.Identity to disable SE)
        :param stride: Output stride
        :param dilation: Dilation factor for 3x3 conv
        :param groups: Number of groups used in convolutions
        :param activation_kwargs: Additional arguments for instantiating activation module.
        :param se_kwargs: Additional arguments for instantiating SE module.
        :param build_residual_branches: Whether to initialize block with already fused paramters (for deployment)
        :param use_residual_connection: Whether to add input x to the output (Enabled in RepVGG, disabled in PP-Yolo)
        :param use_alpha: If True, enables additional learnable weighting parameter for 1x1 branch (PP-Yolo-E Plus)
        """
        super().__init__()

        if activation_kwargs is None:
            activation_kwargs = {}
        if se_kwargs is None:
            se_kwargs = {}

        self.groups = groups
        self.in_channels = in_channels

        self.nonlinearity = activation_type(**activation_kwargs)
        self.se = se_type(**se_kwargs)

        if use_residual_connection and out_channels == in_channels and stride == 1:
            self.no_conv_branch = nn.BatchNorm2d(num_features=in_channels)
        else:
            self.no_conv_branch = None

        self.branch_3x3 = self._conv_bn(
            in_channels=in_channels,
            out_channels=out_channels,
            dilation=dilation,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            groups=groups,
        )
        self.branch_1x1 = self._conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, groups=groups)

        if use_alpha:
            self.alpha = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        else:
            self.alpha = 1

        if not build_residual_branches:
            self.fuse_block_residual_branches()
        else:
            self.build_residual_branches = True

    def forward(self, inputs):
        if not self.build_residual_branches:
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.no_conv_branch is None:
            id_out = 0
        else:
            id_out = self.no_conv_branch(inputs)

        return self.nonlinearity(self.se(self.branch_3x3(inputs) + self.alpha * self.branch_1x1(inputs) + id_out))

    def _get_equivalent_kernel_bias(self):
        """
        Fuses the 3x3, 1x1 and identity branches into a single 3x3 conv layer
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.branch_3x3)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.branch_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.no_conv_branch)
        return kernel3x3 + self.alpha * self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + self.alpha * bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """
        padding the 1x1 convolution weights with zeros to be able to fuse the 3x3 conv layer with the 1x1
        :param kernel1x1: weights of the 1x1 convolution
        :type kernel1x1:
        :return: padded 1x1 weights
        :rtype:
        """
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """
        Fusing of the batchnorm into the conv layer.
        If the branch is the identity branch (no conv) the kernel will simply be eye.
        :param branch:
        :type branch:
        :return:
        :rtype:
        """
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_block_residual_branches(self):
        """
        converts a repvgg block from training model (with branches) to deployment mode (vgg like model)
        :return:
        :rtype:
        """
        if hasattr(self, "build_residual_branches") and not self.build_residual_branches:
            return
        kernel, bias = self._get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.branch_3x3.conv.in_channels,
            out_channels=self.branch_3x3.conv.out_channels,
            kernel_size=self.branch_3x3.conv.kernel_size,
            stride=self.branch_3x3.conv.stride,
            padding=self.branch_3x3.conv.padding,
            dilation=self.branch_3x3.conv.dilation,
            groups=self.branch_3x3.conv.groups,
            bias=True,
        )
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("branch_3x3")
        self.__delattr__("branch_1x1")
        if hasattr(self, "no_conv_branch"):
            self.__delattr__("no_conv_branch")
        if hasattr(self, "alpha"):
            self.__delattr__("alpha")
        self.build_residual_branches = False

    @staticmethod
    def _conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1, dilation=1):
        result = nn.Sequential()
        result.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
                dilation=dilation,
            ),
        )
        result.add_module("bn", nn.BatchNorm2d(num_features=out_channels,eps=1e-03))
        return result

    def prep_model_for_conversion(self, input_size: Optional[Union[tuple, list]] = None, **kwargs):
        self.fuse_block_residual_branches()
        
        

class RepBottleneck(Bottleneck):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=1,activation_type= nn.SiLU,attention=False):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__(c1, c2, shortcut, g=g, e=e)
        c_ = int(c2 * e)  # hidden channels
        if attention:
            self.cv1 = RepVGGBlock(c1,c_,stride=1,groups=g,
                                   activation_type=activation_type,
                                   se_type=LCA,se_kwargs={"channels":c1},
                                   use_residual_connection=False)
            self.cv2 = RepVGGBlock(c_, c2, stride=1, groups=g,
                                    activation_type=activation_type,
                                   se_type=LCA,se_kwargs={"channels":c1},
                                   use_residual_connection=False)
        else:
            self.cv1 = RepVGGBlock(c1,c_,stride=1,groups=g,
                                   activation_type=activation_type,
                                   use_residual_connection=False)
            self.cv2 = RepVGGBlock(c_, c2, stride=1, groups=g,
                                   activation_type=activation_type,
                                   use_residual_connection=False)
class RepBottleneckSG(Bottleneck):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=4, e=1,activation_type= nn.SiLU,attention=False):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__(c1, c2, shortcut, g=g, e=e)
        c_ = int(c2 * e)  # hidden channels
        if attention:
            self.cv1 = RepVGGBlock(c1,c_,stride=1,groups=1,
                                   activation_type=activation_type,
                                   use_residual_connection=False)
            self.shuffle = ShuffleBlock(g)
            self.cv2 = RepVGGBlock(c_, c2, stride=1, groups=g,
                                    activation_type=activation_type,
                                   se_type=LCA,se_kwargs={"channels":c1},
                                   use_residual_connection=False)
        else:
            self.cv1 = RepVGGBlock(c1,c_,stride=1,groups=g,
                                   activation_type=activation_type,
                                   use_residual_connection=False)
            self.shuffle = ShuffleBlock(g)
            self.cv2 = RepVGGBlock(c_, c2, stride=1, groups=g,
                                   activation_type=activation_type,
                                   use_residual_connection=False)
    def forward(self, x):
        return x + self.cv2(self.shuffle(self.cv1(x))) if self.add else self.cv2((self.shuffle(self.cv1(x))))
class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups
    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N,C,H,W = x.size()
        g = self.groups
        return x.view(N,g,C//g,H,W).permute(0,2,1,3,4).reshape(N,C,H,W)
        
class RepC2f(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c = int(c2 * e)  # hidden channels
        self.m = nn.ModuleList(RepBottleneck(self.c, self.c,shortcut,g=g) for _ in range(n))

class RepC2fSG(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=4, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c = int(c2 * e)  # hidden channels
        self.m = nn.ModuleList(RepBottleneckSG(self.c, self.c,shortcut,g=g,attention=True) for _ in range(n))


def fuse_conv_and_bn(conv, bn):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)
    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))
    # prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    return fusedconv
    
    