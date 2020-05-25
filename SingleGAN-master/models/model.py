import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.parallel
from torch.autograd import Variable
from models.densenet import densenet121
#from models.pretrained_densenet_efficient  import pretrained_efficient_densenet169
#import torch.utils.model_zoo as model_zoo
#import torch.nn.functional as F
import math
import pytorch_msssim

import functools
from .cbin import CBINorm2d
from .cbbn import CBBNorm2d
from torch.nn.utils.spectral_norm import spectral_norm

def mssim_loss(img1,img2):
   # value = torch.stack([pytorch_msssim.msssim(i1, i2) for i1, i2 in zip(img1, img2)], dim=0)
    value = pytorch_msssim.msssim(img1, img2)
    return torch.add(1, -value)
def get_norm_layer(layer_type='instance', num_con=2):
    if layer_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        c_norm_layer = functools.partial(CBBNorm2d, affine=True, num_con=num_con)
    elif layer_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        c_norm_layer = functools.partial(CBINorm2d, affine=True, num_con=num_con)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % layer_type)
    return norm_layer, c_norm_layer

def get_nl_layer(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif layer_type == 'sigmoid':
        nl_layer = nn.Sigmoid
    elif layer_type == 'tanh':
        nl_layer = nn.Tanh
    else:
        raise NotImplementedError('nl_layer layer [%s] is not found' % layer_type)
    return nl_layer
def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>
def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        print('cuda is adding')
        print(gpu_ids)
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type)
    return net
def set_requires_grad(nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
def define_net(net_type,input_nc=3, output_nc=3, ngf=64,nc=4,ndf=64,block_num=3, e_blocks=6,norm_type='normal',gpu_ids=[]):
    if net_type=='generator':
       net=SingleGeneratorSkip(input_nc=input_nc, output_nc=input_nc, ngf=ngf,
                        nc=nc, e_blocks=e_blocks, norm_type=norm_type)
    elif net_type=='discriminator':
        net=D_NET_Multi(input_nc=input_nc, ndf=ndf, block_num=block_num,norm_type=norm_type)
    elif net_type== 'D_NET_Multi_Classify':
        net = D_NET_Multi_Classify(input_nc=input_nc, ndf=ndf, block_num=block_num, norm_type=norm_type)
    elif net_type=='classify' :
        net = densenet121(pretrained=True)
        set_requires_grad(net,False)
        num_ftrs = net.classifier.in_features
        net.classifier = nn.Linear(num_ftrs, 2)
    elif net_type=='generator_simple':
        net = SingleGeneratorSimple(input_nc=input_nc, output_nc=input_nc, ngf=ngf,
                                  nc=nc, e_blocks=e_blocks, norm_type=norm_type)
    elif net_type=='generator_classify':
        net = SingleGeneratorClassify(input_nc=input_nc, output_nc=input_nc, ngf=ngf,
                                  nc=nc, e_blocks=e_blocks, norm_type=norm_type)
    elif net_type == 'muti_discriminator':
        net = D_NLayersMulti(input_nc=input_nc, ndf=ngf, norm_type=norm_type)
    return init_net(net,init_type='normal', init_gain=0.02,gpu_ids=gpu_ids)
    # if len(gpu_ids) > 0:
    #     assert (torch.cuda.is_available())
    #     net.to(gpu_ids[0])
    #     net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    # return net
def weights_init(init_type='xavier'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'normal':
                init.normal(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant(m.bias.data, 0.0)
    return init_fun    
    
class Conv2dBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=0, pad_type='reflect', bias=True, norm_layer=None, nl_layer=None):
        super(Conv2dBlock, self).__init__()
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        self.conv = spectral_norm(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=0, bias=bias))
        if norm_layer is not None:
            self.norm = norm_layer(out_planes)
        else:
            self.norm = lambda x: x
        
        if nl_layer is not None:
            self.activation = nl_layer()
        else:
            self.activation = lambda x: x
                     
    def forward(self, x):
        return self.activation(self.norm(self.conv(self.pad(x))))

class TrConv2dBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=0, bias=True, dilation=1, norm_layer=None, nl_layer=None):
        super(TrConv2dBlock, self).__init__()
        self.trConv = spectral_norm(nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, bias=bias, dilation=dilation))
        if norm_layer is not None:
            self.norm = norm_layer(out_planes)
        else:
            self.norm = lambda x: x
        
        if nl_layer is not None:
            self.activation = nl_layer()
        else:
            self.activation = lambda x: x
                     
    def forward(self, x):
        return self.activation(self.norm(self.trConv(x)))

class Upsampling2dBlock(nn.Module):
    def __init__(self, in_planes, out_planes, type='Trp', norm_layer=None, nl_layer=None):
        super(Upsampling2dBlock, self).__init__()
        if type=='Trp':
            self.upsample = TrConv2dBlock(in_planes,out_planes,kernel_size=4,stride=2,padding=1,bias=False,norm_layer=norm_layer,nl_layer=nl_layer)
        elif type=='Ner':
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                Conv2dBlock(in_planes,out_planes,kernel_size=4, stride=1, padding=1, pad_type='reflect', bias=False,norm_layer=norm_layer,nl_layer=nl_layer)
                )
        else:
            raise('None Upsampling type {}'.format(type))
    def forward(self, x):
        return self.upsample(x)
    
def conv3x3(in_planes, out_planes, norm_layer=None, nl_layer=None):
    "3x3 convolution with padding"
    return Conv2dBlock(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, pad_type='reflect', bias=False, norm_layer=norm_layer, nl_layer=nl_layer)                     




################ Generator ###################       
class CResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, h_dim, c_norm_layer=None, nl_layer=None):
        super(CResidualBlock, self).__init__()
        self.c1 = Conv2dBlock(h_dim,h_dim, kernel_size=3, stride=1, padding=1, pad_type='reflect', bias=False)
        self.n1 = c_norm_layer(h_dim)
        self.l1 = nl_layer()
        self.c2 = Conv2dBlock(h_dim,h_dim, kernel_size=3, stride=1, padding=1, pad_type='reflect', bias=False)
        self.n2 = c_norm_layer(h_dim)

    def forward(self, input):
        x, c = input[0], input[1]
        y = self.l1(self.n1(self.c1(x),c))
        y = self.n2(self.c2(y),c)
        return [x + y,  c]
class SingleGeneratorSimple(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, nc=2, e_blocks=6, norm_type='instance', up_type='Trp'
                 ,pad_type='zero'):
        super(SingleGeneratorSimple, self).__init__()
        norm_layer, c_norm_layer = get_norm_layer(layer_type=norm_type, num_con=nc)
        nl_layer = get_nl_layer(layer_type='relu')
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.c1 = nn.Sequential(nn.ReflectionPad2d(3),
                                  nn.Conv2d(input_nc, ngf, kernel_size=7,padding=0, bias=use_bias))
        self.n1 = c_norm_layer(ngf)#,
                                #nn.ReLU(True))
        self.a1 = nl_layer()

        self.d1 = Conv2dBlock(ngf, ngf*2, kernel_size=3, stride=2, padding=1, pad_type=pad_type, bias=use_bias)
        self.d1_n = c_norm_layer(ngf*2)
        self.d1_a = nl_layer()
        down_block_1 = []
        for i in range(e_blocks):
            down_block_1.append(CResidualBlock(ngf*2, c_norm_layer=c_norm_layer, nl_layer=nl_layer))
        self.down_block_1 = nn.Sequential(*down_block_1)
        self.d2 = Conv2dBlock(ngf*2, ngf*4, kernel_size=3, stride=2, padding=1, pad_type=pad_type, bias=use_bias)
        self.d2_n = c_norm_layer(ngf*4)  # ,
        # nn.ReLU(True))
        self.d2_a = nl_layer()
        block = []
        for i in range(e_blocks+3):
            block.append(CResidualBlock(ngf*4, c_norm_layer=c_norm_layer,nl_layer=nl_layer))
        self.resBlocks =  nn.Sequential(*block)
        self.upsample2 = nn.Sequential(nn.ConvTranspose2d(ngf * 4, ngf * 2,
                                                          kernel_size=3, stride=2,
                                                          padding=1, output_padding=1,
                                                          bias=use_bias),
                                       norm_layer(ngf * 2),
                                       nn.ReLU(True))
        up_block1 = []
        for i in range(e_blocks):  # add ResNet blocks
            up_block1 += [
                ResnetBlock(ngf * 2, padding_type=pad_type, norm_layer=norm_layer, use_dropout=False,
                            use_bias=use_bias)]
        self.resnet_up_Block1 = nn.Sequential(*up_block1)
        self.upsample1 = nn.Sequential(nn.ConvTranspose2d(ngf * 2, ngf,
                                                          kernel_size=3, stride=2,
                                                          padding=1, output_padding=1,
                                                          bias=use_bias),
                                       norm_layer(ngf * 2),
                                       nn.ReLU(True))

        self.out = nn.Sequential(nn.ReflectionPad2d(3),
                                 nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                                 nn.Tanh())

    def forward(self, x, c):
       # print('model_simple')
        x =  self.a1(self.n1(self.c1(x), c))
        down0_1 = self.down_block_1([self.d1_a(self.d1_n(self.d1(x), c)),c])
        down1_2 = self.d2_a(self.d2_n(self.d2(down0_1[0]), c))
        resblock = self.resBlocks([down1_2, c])
        up1_0 = self.upsample2(resblock[0])
        up_0 =  self.resnet_up_Block1(up1_0 + down0_1[0])
        out=self.upsample1(up_0)
        y=self.out(out)
        return y
class SingleGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, nc=2, e_blocks=6, norm_type='instance', up_type='Trp'
                 ,pad_type='zero'):
        super(SingleGenerator, self).__init__()
        norm_layer, c_norm_layer = get_norm_layer(layer_type=norm_type, num_con=nc)
        nl_layer = get_nl_layer(layer_type='relu')
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.c1 = nn.Sequential(nn.ReflectionPad2d(3),
                                  nn.Conv2d(input_nc, ngf, kernel_size=7,padding=0, bias=use_bias))
        #self.c1 = Conv2dBlock(input_nc, ngf, kernel_size=7, stride=1, padding=3, pad_type=pad_type, bias=use_bias)
        self.n1 = c_norm_layer(ngf)#,
                                #nn.ReLU(True))
        self.a1 = nl_layer()

        self.d1 = Conv2dBlock(ngf, ngf*2, kernel_size=3, stride=2, padding=1, pad_type=pad_type, bias=use_bias)
        self.d1_n = c_norm_layer(ngf*2)  # ,
        # nn.ReLU(True))
        self.d1_a = nl_layer()
        # self.n2 = c_norm_layer(ngf*2)
        # self.a2 = nl_layer()
        down_block_1 = []
        for i in range(e_blocks):
            down_block_1.append(CResidualBlock(ngf*2, c_norm_layer=c_norm_layer, nl_layer=nl_layer))
        self.down_block_1 = nn.Sequential(*down_block_1)


        self.d2 = Conv2dBlock(ngf*2, ngf*4, kernel_size=3, stride=2, padding=1, pad_type=pad_type, bias=use_bias)
        self.d2_n = c_norm_layer(ngf*4)  # ,
        # nn.ReLU(True))
        self.d2_a = nl_layer()
        # self.n3 = c_norm_layer(ngf*4)
        # self.a3 = nl_layer()
        down_block_2 = []
        for i in range(e_blocks):
            down_block_2.append(CResidualBlock(ngf * 4, c_norm_layer=c_norm_layer, nl_layer=nl_layer))
        self.down_block_2 = nn.Sequential(*down_block_2)



        self.d3 = Conv2dBlock(ngf * 4, ngf * 8, kernel_size=3, stride=2, padding=1, pad_type=pad_type, bias=use_bias)
        self.d3_n = c_norm_layer(ngf*8)  # ,
        # nn.ReLU(True))
        self.d3_a = nl_layer()

        block = []
        for i in range(9):
            block.append(CResidualBlock(ngf*8, c_norm_layer=c_norm_layer,nl_layer=nl_layer))
        self.resBlocks =  nn.Sequential(*block)



        self.upsample3 = nn.Sequential(nn.ConvTranspose2d(ngf * 8, ngf*4,
                                                          kernel_size=3, stride=2,
                                                          padding=1, output_padding=1,
                                                          bias=use_bias),
                                       norm_layer(ngf*4),
                                       nn.ReLU(True))
        up_block2= []
        for i in range(3):  # add ResNet blocks
            up_block2 += [
                ResnetBlock(ngf * 4, padding_type=pad_type, norm_layer=norm_layer, use_dropout=False,
                            use_bias=use_bias)]
        self.resnet_up_Block2 = nn.Sequential(*up_block2)


        self.upsample2 = nn.Sequential(nn.ConvTranspose2d(ngf * 4, ngf * 2,
                                                          kernel_size=3, stride=2,
                                                          padding=1, output_padding=1,
                                                          bias=use_bias),
                                       norm_layer(ngf * 2),
                                       nn.ReLU(True))
        up_block1 = []
        for i in range(3):  # add ResNet blocks
            up_block1 += [
                ResnetBlock(ngf * 2, padding_type=pad_type, norm_layer=norm_layer, use_dropout=False,
                            use_bias=use_bias)]
        self.resnet_up_Block1 = nn.Sequential(*up_block1)
        self.upsample1 = nn.Sequential(nn.ConvTranspose2d(ngf * 2, ngf,
                                                          kernel_size=3, stride=2,
                                                          padding=1, output_padding=1,
                                                          bias=use_bias),
                                       norm_layer(ngf * 2),
                                       nn.ReLU(True))

        self.out = nn.Sequential(nn.ReflectionPad2d(3),
                                 nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                                 nn.Tanh())
        self.up4 = nn.ConvTranspose2d(ngf * 4, ngf * 8,
                                                          kernel_size=4, stride=2,
                                                          padding=1, output_padding=1,
                                                          bias=use_bias)

        #self.up_n4 = nn.Sequential(c_norm_layer(ngf * 8),
        #                         nn.ReLU(True))
        #
        # block = [Upsampling2dBlock(ngf*8,ngf*4,type=up_type,norm_layer=norm_layer,nl_layer=nl_layer)]
        #
        # block += [Upsampling2dBlock(ngf*2,ngf,type=up_type,norm_layer=norm_layer,nl_layer=nl_layer)]
        #
        # block +=[Conv2dBlock(ngf, output_nc, kernel_size=7, stride=1, padding=3, pad_type='reflect', bias=use_bias,nl_layer=nn.Tanh)]
        # self.upBlocks = nn.Sequential(*block)

    def forward(self, x, c):
        # x = self.a1(self.n1(self.c1(x),c))
        # x = self.a2(self.n2(self.c2(x),c))
        # x = self.a3(self.n3(self.c3(x),c))
        # x = self.resBlocks([x,c])[0]
        # y = self.upBlocks(x)
        # self.c1(x),
        # self.down_block_1
     #   print(x.shape)
        x =  self.a1(self.n1(self.c1(x), c))
        down0_1 = self.down_block_1([self.d1_a(self.d1_n(self.d1(x), c)),c])

        down1_2 = self.down_block_2([self.d2_a(self.d2_n(self.d2(down0_1[0]),c)),c] )
        down2_3 = self.d3_a(self.d3_n(self.d3(down1_2[0]),c))

        resblock = self.resBlocks([down2_3,c])

        up3_2 = self.upsample3(resblock[0])
        up2_1 = self.resnet_up_Block2(up3_2 + down1_2[0])
        up1_0 = self.upsample2(up2_1)
        up_0 =  self.resnet_up_Block1(up1_0 + down0_1[0])
        out=self.upsample1(up_0)
        y=self.out(out)
        # x = self.resBlocks([x, c])[0]
        # y = self.upBlocks(x)
        # self.c1(x),

        return y
class SingleGeneratorSkip(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, nc=2, e_blocks=6, norm_type='instance', up_type='Trp'
                 ,pad_type='zero'):
        super(SingleGeneratorSkip, self).__init__()
        norm_layer, c_norm_layer = get_norm_layer(layer_type=norm_type, num_con=nc)
        nl_layer = get_nl_layer(layer_type='relu')
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.c1 = nn.Sequential(nn.ReflectionPad2d(3),
                                  nn.Conv2d(input_nc, ngf, kernel_size=7,padding=0, bias=use_bias))
        #self.c1 = Conv2dBlock(input_nc, ngf, kernel_size=7, stride=1, padding=3, pad_type=pad_type, bias=use_bias)
        self.n1 = c_norm_layer(ngf)#,
                                #nn.ReLU(True))
        self.a1 = nl_layer()

        self.d1 = Conv2dBlock(ngf, ngf*2, kernel_size=3, stride=2, padding=1, pad_type=pad_type, bias=use_bias)
        self.d1_n = c_norm_layer(ngf*2)
        self.d1_a = nl_layer()
        down_block_1 = []
        for i in range(e_blocks):
            down_block_1.append(CResidualBlock(ngf*2, c_norm_layer=c_norm_layer, nl_layer=nl_layer))
        self.down_block_1 = nn.Sequential(*down_block_1)


        self.d2 = Conv2dBlock(ngf*2, ngf*4, kernel_size=3, stride=2, padding=1, pad_type=pad_type, bias=use_bias)
        self.d2_n = c_norm_layer(ngf*4)  # ,
        # nn.ReLU(True))
        self.d2_a = nl_layer()
        # self.n3 = c_norm_layer(ngf*4)
        # self.a3 = nl_layer()
        down_block_2 = []
        for i in range(e_blocks):
            down_block_2.append(CResidualBlock(ngf * 4, c_norm_layer=c_norm_layer, nl_layer=nl_layer))
        self.down_block_2 = nn.Sequential(*down_block_2)



        self.d3 = Conv2dBlock(ngf * 4, ngf * 8, kernel_size=3, stride=2, padding=1, pad_type=pad_type, bias=use_bias)
        self.d3_n = c_norm_layer(ngf*8)  # ,
        # nn.ReLU(True))
        self.d3_a = nl_layer()

        block = []
        for i in range(9):
            block.append(CResidualBlock(ngf*8, c_norm_layer=c_norm_layer,nl_layer=nl_layer))
        self.resBlocks =  nn.Sequential(*block)

        self.upsample3 = nn.Sequential(nn.ConvTranspose2d(ngf * 8, ngf*4,
                                                          kernel_size=3, stride=2,
                                                          padding=1, output_padding=1,
                                                          bias=use_bias),
                                       norm_layer(ngf*4),
                                       nn.ReLU(True))
        up_block2= []
        for i in range(3):  # add ResNet blocks
            up_block2 += [
                ResnetBlock(ngf * 4, padding_type=pad_type, norm_layer=norm_layer, use_dropout=False,
                            use_bias=use_bias)]
        self.resnet_up_Block2 = nn.Sequential(*up_block2)


        self.upsample2 = nn.Sequential(nn.ConvTranspose2d(ngf * 4, ngf * 2,
                                                          kernel_size=3, stride=2,
                                                          padding=1, output_padding=1,
                                                          bias=use_bias),
                                       norm_layer(ngf * 2),
                                       nn.ReLU(True))
        up_block1 = []
        for i in range(3):  # add ResNet blocks
            up_block1 += [
                ResnetBlock(ngf * 2, padding_type=pad_type, norm_layer=norm_layer, use_dropout=False,
                            use_bias=use_bias)]
        self.resnet_up_Block1 = nn.Sequential(*up_block1)
        self.upsample1 = nn.Sequential(nn.ConvTranspose2d(ngf * 2, ngf,
                                                          kernel_size=3, stride=2,
                                                          padding=1, output_padding=1,
                                                          bias=use_bias),
                                       norm_layer(ngf * 2),
                                       nn.ReLU(True))

        self.out = nn.Sequential(nn.ReflectionPad2d(3),
                                 nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                                 nn.Tanh())

    def forward(self, x, c):

        x =  self.a1(self.n1(self.c1(x), c))
        down0_1 = self.down_block_1([self.d1_a(self.d1_n(self.d1(x), c)),c])

        down1_2 = self.down_block_2([self.d2_a(self.d2_n(self.d2(down0_1[0]),c)),c] )
        down2_3 = self.d3_a(self.d3_n(self.d3(down1_2[0]),c))

        resblock = self.resBlocks([down2_3,c])

        up3_2 = self.upsample3(resblock[0])
        up2_1 = self.resnet_up_Block2(up3_2 + down1_2[0])
        up1_0 = self.upsample2(up2_1)
        up_0 =  self.resnet_up_Block1(up1_0 + down0_1[0])
        out=self.upsample1(up_0)
        y=self.out(out)


        return y
class SingleGeneratorClassify(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, nc=2, e_blocks=6, norm_type='instance', up_type='Trp'
                 ,pad_type='zero'):
        super(SingleGeneratorClassify, self).__init__()
        norm_layer, c_norm_layer = get_norm_layer(layer_type=norm_type, num_con=nc)
        nl_layer = get_nl_layer(layer_type='relu')
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.c1 = nn.Sequential(nn.ReflectionPad2d(3),
                                  nn.Conv2d(input_nc, ngf, kernel_size=7,padding=0, bias=use_bias))
        #self.c1 = Conv2dBlock(input_nc, ngf, kernel_size=7, stride=1, padding=3, pad_type=pad_type, bias=use_bias)
        self.n1 = c_norm_layer(ngf)#,
                                #nn.ReLU(True))
        self.a1 = nl_layer()

        self.d1 = Conv2dBlock(ngf, ngf*2, kernel_size=3, stride=2, padding=1, pad_type=pad_type, bias=use_bias)
        self.d1_n = c_norm_layer(ngf*2)
        self.d1_a = nl_layer()
        down_block_1 = []
        for i in range(3):
            down_block_1.append(CResidualBlock(ngf*2, c_norm_layer=c_norm_layer, nl_layer=nl_layer))
        self.down_block_1 = nn.Sequential(*down_block_1)


        self.d2 = Conv2dBlock(ngf*2, ngf*4, kernel_size=3, stride=2, padding=1, pad_type=pad_type, bias=use_bias)
        self.d2_n = c_norm_layer(ngf*4)  # ,
        # nn.ReLU(True))
        self.d2_a = nl_layer()
        # self.n3 = c_norm_layer(ngf*4)
        # self.a3 = nl_layer()
        down_block_2 = []
        for i in range(3):
            down_block_2.append(CResidualBlock(ngf * 4, c_norm_layer=c_norm_layer, nl_layer=nl_layer))
        self.down_block_2 = nn.Sequential(*down_block_2)



        self.d3 = Conv2dBlock(ngf * 4, ngf * 8, kernel_size=3, stride=2, padding=1, pad_type=pad_type, bias=use_bias)
        self.d3_n = c_norm_layer(ngf*8)  # ,
        # nn.ReLU(True))
        self.d3_a = nl_layer()

        block = []
        for i in range(9):
            block.append(CResidualBlock(ngf*8, c_norm_layer=c_norm_layer,nl_layer=nl_layer))
        self.resBlocks =  nn.Sequential(*block)

        self.upsample3 = nn.Sequential(nn.ConvTranspose2d(ngf * 8, ngf*4,
                                                          kernel_size=3, stride=2,
                                                          padding=1, output_padding=1,
                                                          bias=use_bias),
                                       norm_layer(ngf*4),
                                       nn.ReLU(True))
        up_block2= []
        for i in range(3):  # add ResNet blocks
            up_block2 += [
                ResnetBlock(ngf * 4, padding_type=pad_type, norm_layer=norm_layer, use_dropout=False,
                            use_bias=use_bias)]
        self.resnet_up_Block2 = nn.Sequential(*up_block2)


        self.upsample2 = nn.Sequential(nn.ConvTranspose2d(ngf * 4, ngf * 2,
                                                          kernel_size=3, stride=2,
                                                          padding=1, output_padding=1,
                                                          bias=use_bias),
                                       norm_layer(ngf * 2),
                                       nn.ReLU(True))
        up_block1 = []
        for i in range(3):  # add ResNet blocks
            up_block1 += [
                ResnetBlock(ngf * 2, padding_type=pad_type, norm_layer=norm_layer, use_dropout=False,
                            use_bias=use_bias)]
        self.resnet_up_Block1 = nn.Sequential(*up_block1)
        self.upsample1 = nn.Sequential(nn.ConvTranspose2d(ngf * 2, ngf,
                                                          kernel_size=3, stride=2,
                                                          padding=1, output_padding=1,
                                                          bias=use_bias),
                                       norm_layer(ngf * 2),
                                       nn.ReLU(True))

        self.out = nn.Sequential(nn.ReflectionPad2d(3),
                                 nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                                 nn.Tanh())

    def forward(self, x, c):

        x =  self.a1(self.n1(self.c1(x), c))
        down0_1 = self.down_block_1([self.d1_a(self.d1_n(self.d1(x), c)),c])

        down1_2 = self.down_block_2([self.d2_a(self.d2_n(self.d2(down0_1[0]),c)),c] )
        down2_3 = self.d3_a(self.d3_n(self.d3(down1_2[0]),c))

        resblock = self.resBlocks([down2_3,c])

        up3_2 = self.upsample3(resblock[0])
        up2_1 = self.resnet_up_Block2(up3_2 + down1_2[0])
        up1_0 = self.upsample2(up2_1)
        up_0 =  self.resnet_up_Block1(up1_0 + down0_1[0])
        out=self.upsample1(up_0)
        y=self.out(out)


        return y
# class SingleGeneratorMuti(nn.Module):
#     def __init__(self, input_nc=3, output_nc=3, ngf=64, nc=2, e_blocks=6, norm_type='instance', up_type='Trp'
#                  ,pad_type='zero'):
#         super(SingleGeneratorMuti, self).__init__()
#         norm_layer, c_norm_layer = get_norm_layer(layer_type=norm_type, num_con=nc)
#         nl_layer = get_nl_layer(layer_type='relu')
#         if type(norm_layer) == functools.partial:
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d
#         self.c1 = nn.Sequential(nn.ReflectionPad2d(3),
#                                   nn.Conv2d(input_nc, ngf, kernel_size=7,padding=0, bias=use_bias))
#         #self.c1 = Conv2dBlock(input_nc, ngf, kernel_size=7, stride=1, padding=3, pad_type=pad_type, bias=use_bias)
#         self.n1 = c_norm_layer(ngf)
#         self.a1 = nl_layer()
#
#         self.d1 = Conv2dBlock(ngf, ngf*2, kernel_size=3, stride=2, padding=1, pad_type=pad_type, bias=use_bias)
#         self.d1_n = c_norm_layer(ngf*2)
#         self.d1_a = nl_layer()
#
#         down_block_1 = []
#         for i in range(e_blocks):
#             down_block_1.append(CResidualBlock(ngf*2, c_norm_layer=c_norm_layer, nl_layer=nl_layer))
#         self.down_block_1 = nn.Sequential(*down_block_1)
#         self.d2 = Conv2dBlock(ngf*2, ngf*4, kernel_size=3, stride=2, padding=1, pad_type=pad_type, bias=use_bias)
#         self.d2_n = c_norm_layer(ngf*4)
#         self.d2_a = nl_layer()
#
#
#         block = []
#         for i in range(9):
#             block.append(CResidualBlock(ngf*4, c_norm_layer=c_norm_layer,nl_layer=nl_layer))
#         self.resBlocks =  nn.Sequential(*block)
#         self.upsample2 = nn.Sequential(nn.ConvTranspose2d(ngf * 4, ngf*2,
#                                                           kernel_size=3, stride=2,
#                                                           padding=1, output_padding=1,
#                                                           bias=use_bias),
#                                        norm_layer(ngf*4),
#                                        nn.ReLU(True))
#         up_block1= []
#         for i in range(3):  # add ResNet blocks
#             up_block1 += [
#                 ResnetBlock(ngf * 2, padding_type=pad_type, norm_layer=norm_layer, use_dropout=False,
#                             use_bias=use_bias)]
#         self.up_block1 = nn.Sequential(*up_block1)
#
#
#         self.upsample1 = nn.Sequential(nn.ConvTranspose2d(ngf * 2, ngf,
#                                                           kernel_size=3, stride=2,
#                                                           padding=1, output_padding=1,
#                                                           bias=use_bias),
#                                        norm_layer(ngf * 2),
#                                        nn.ReLU(True))
#         self.out = nn.Sequential(nn.ReflectionPad2d(3),
#                                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
#                                  nn.Tanh())
#
#
#     def forward(self, x, c):
#
#         x =  self.a1(self.n1(self.c1(x), c))
#         down0_1 = self.down_block_1([self.d1_a(self.d1_n(self.d1(x), c)),c])
#
#         down1_2 = self.d2_a(self.d2_n(self.d2(down0_1[0]),c))
#         resblock = self.resBlocks([down1_2,c])
#
#         up2_1 = self.upsample2(resblock[0])
#         up1_0 = self.up_block1(up2_1 + down0_1[0])
#         out = self.upsample1(up1_0)
#         y=self.out(out)
#         return y
class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


################ Discriminator ##########################
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
class D_NET(nn.Module):
    def __init__(self, input_nc=3, ndf=32, block_num=3,  norm_type='instance'):
        super(D_NET, self).__init__()
        nl_layer = get_nl_layer('lrelu')
        block = [Conv2dBlock(input_nc, ndf, kernel_size=4,stride=2,padding=1,bias=False,nl_layer=nl_layer)]
        dim_in=ndf
        for n in range(1, block_num):
            dim_out = min(dim_in*2, ndf*8)
            block += [Conv2dBlock(dim_in, dim_out, kernel_size=4, stride=2, padding=1,bias=False,nl_layer=nl_layer)]
            dim_in = dim_out
        dim_out = min(dim_in*2, ndf*8)
        block += [Conv2dBlock(dim_in, 1, kernel_size=4, stride=1, padding=1,bias=True) ]
        self.conv = nn.Sequential(*block)

    def forward(self, x):
        return self.conv(x)
        

class D_NET_Multi_Classify(nn.Module):
    def __init__(self, input_nc=3, ndf=32, block_num=3, norm_type='instance'):
        super(D_NET_Multi_Classify, self).__init__()
        self.numD=3
        norm_layer,cc=get_norm_layer(norm_type,2)
        self.model_0= NLayerDiscriminator(input_nc=input_nc, ndf=ndf, n_layers=block_num, norm_layer=norm_layer)
       # self.model_1 = D_NET(input_nc=input_nc, ndf=ndf, block_num=block_num, norm_type=norm_type)
        for i in range(1,self.numD):
            netD = NLayerDiscriminator(input_nc=input_nc, ndf=ndf, n_layers=block_num-1, norm_layer=norm_layer)
            setattr(self, 'model_' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        # self.model_2 = NLayerDiscriminator(input_nc=input_nc, ndf=ndf, n_layers=block_num-1, norm_layer=norm_layer)
        # self.model_3 = NLayerDiscriminator(input_nc=input_nc, ndf=ndf, n_layers=block_num-1, norm_layer=norm_layer)
        # self.model_4 = NLayerDiscriminator(input_nc=input_nc, ndf=ndf, n_layers=block_num-1, norm_layer=norm_layer)

    # self.model_2 = D_NET(input_nc=input_nc, ndf=ndf//2, block_num=block_num, norm_type=norm_type)
    def singleD_forward(self, model, input):
            return [model(input)]
    def forward(self, x):
        result = []
        input_downsampled = x
        for i in range(self.numD):
            model = getattr(self, 'model_' + str(i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (self.numD - 1):
                input_downsampled = self.downsample(input_downsampled)

        return result
class D_NET_Multi(nn.Module):
    def __init__(self, input_nc=3, ndf=32, block_num=3, norm_type='instance'):
        super(D_NET_Multi, self).__init__()
        self.numD=4

        #n_layers = block_num-1
        norm_layer,cc=get_norm_layer(norm_type,2)
        self.model_0= NLayerDiscriminator(input_nc=input_nc, ndf=ndf, n_layers=block_num, norm_layer=norm_layer)
       # self.model_1 = D_NET(input_nc=input_nc, ndf=ndf, block_num=block_num, norm_type=norm_type)
        for i in range(1,self.numD):
            netD = NLayerDiscriminator(input_nc=input_nc, ndf=ndf, n_layers=block_num-1, norm_layer=norm_layer)
            setattr(self, 'model_' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        # self.model_2 = NLayerDiscriminator(input_nc=input_nc, ndf=ndf, n_layers=block_num-1, norm_layer=norm_layer)
        # self.model_3 = NLayerDiscriminator(input_nc=input_nc, ndf=ndf, n_layers=block_num-1, norm_layer=norm_layer)
        # self.model_4 = NLayerDiscriminator(input_nc=input_nc, ndf=ndf, n_layers=block_num-1, norm_layer=norm_layer)

    # self.model_2 = D_NET(input_nc=input_nc, ndf=ndf//2, block_num=block_num, norm_type=norm_type)
    def singleD_forward(self, model, input):
            return [model(input)]
    def forward(self, x):
        result = []
        input_downsampled = x
        for i in range(self.numD):
            model = getattr(self, 'model_' + str(i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (self.numD - 1):
                input_downsampled = self.downsample(input_downsampled)

        return result



class D_NLayersMulti(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3,
                 norm_type='instance', num_D=3):
        super(D_NLayersMulti, self).__init__()
        norm_layer, cc = get_norm_layer(norm_type, 2)
        # st()
        self.num_D = num_D
        if num_D == 1:
            layers = self.get_layers(input_nc, ndf, n_layers, norm_layer)
            self.model = nn.Sequential(*layers)
        else:
            layers = self.get_layers(input_nc, ndf, n_layers, norm_layer)
            self.add_module("model_0", nn.Sequential(*layers))
            self.down = nn.AvgPool2d(3, stride=2, padding=[
                                     1, 1], count_include_pad=False)
            for i in range(0, num_D):
                ndf_i = int(round(ndf / (2**i)))
                layers = self.get_layers(input_nc, ndf_i, n_layers, norm_layer)
                self.add_module("model_%d" % i, nn.Sequential(*layers))

    def get_layers(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw,
                              stride=2, padding=padw), nn.LeakyReLU(0.2, True)]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1,
                               kernel_size=kw, stride=1, padding=padw)]

        return sequence

    def forward(self, input):
        if self.num_D == 1:
            return self.model(input)
        result = []
        down = input
        for i in range(self.num_D):
            model = getattr(self, "model_%d" % i)
            result.append(model(down))
            if i != self.num_D - 1:
                down = self.down(down)
        return result
################ Encoder ##################################
def meanpoolConv(inplanes, outplanes):
    sequence = []
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    sequence += [Conv2dBlock(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=True)]
    return nn.Sequential(*sequence)
    
def convMeanpool(inplanes, outplanes):
    sequence = []
    sequence += [conv3x3(inplanes, outplanes)]
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*sequence)
    
class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, c_norm_layer=None, nl_layer=None):
        super(BasicBlock, self).__init__()
        self.cnorm1 = c_norm_layer(inplanes)
        self.nl1 = nl_layer()
        self.conv1 = conv3x3(inplanes, inplanes)
        self.cnorm2 = c_norm_layer(inplanes)
        self.nl2 = nl_layer()
        self.cmp = convMeanpool(inplanes, outplanes)
        self.shortcut = meanpoolConv(inplanes, outplanes)

    def forward(self, input):
        x, d = input
        out = self.cmp(self.nl2(self.cnorm2(self.conv1(self.nl1(self.cnorm1(x,d))),d)))
        out = out + self.shortcut(x)
        return [out,d]
        
class Encoder(nn.Module):
    def __init__(self, input_nc=3, output_nc=1, nef=64, nd=2, n_blocks=4, norm_type='instance'):
        # img 128*128 -> n_blocks=4 // img 256*256 -> n_blocks=5 
        super(Encoder, self).__init__()
        _, c_norm_layer = get_norm_layer(layer_type=norm_type, num_con=nd)
        max_ndf = 4
        nl_layer = get_nl_layer(layer_type='lrelu')
        self.entry = Conv2dBlock(input_nc, nef, kernel_size=4, stride=2, padding=1, bias=True)
        conv_layers =[]
        for n in range(1, n_blocks):
            input_ndf = nef * min(max_ndf, n)  # 2**(n-1)
            output_ndf = nef * min(max_ndf, n+1)  # 2**n
            conv_layers += [BasicBlock(input_ndf, output_ndf, c_norm_layer, nl_layer)]
        self.middle = nn.Sequential(*conv_layers)
        self.exit = nn.Sequential(*[nl_layer(), nn.AdaptiveAvgPool2d(1)])
        
        self.fc = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        self.fcVar = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
        
    def forward(self, x, d):
        x_conv = self.exit(self.middle([self.entry(x),d])[0])
        b = x_conv.size(0)
        x_conv = x_conv.view(b, -1)
        mu = self.fc(x_conv)
        logvar = self.fcVar(x_conv)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar