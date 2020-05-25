from __future__ import print_function
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torchvision.utils import make_grid
from models.model import D_NET_Multi, SingleGenerator, Encoder, weights_init, define_net, mssim_loss
from util.loss import GANLoss, KL_loss
from util.util import tensor2im, tensor2imMy
import numpy as np


################## SingleGAN #############################
class SingleGAN():
    def name(self):
        return 'SingleGAN'

    def initialize(self, opt):
        self.gpu_ids = [0, 1, 2, 3]
        if not opt.cpu:
            self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        else:
            self.device = torch.device('cpu')
        torch.cuda.set_device(self.device)
        # torch.cuda.set_device(opt.gpu)
        cudnn.benchmark = True
        self.opt = opt
        self.build_models()

    def build_models(self):
        ################### generator #########################################
        self.model_names = ['G', 'Ds']
        # self.G = define_net(net_type='generator',input_nc=self.opt.input_nc, output_nc=self.opt.input_nc, ngf=self.opt.ngf, nc=self.opt.c_num+self.opt.d_num, e_blocks=self.opt.e_blocks, norm_type=self.opt.norm,gpu_ids=self.gpu_ids)
        if self.opt.classify_mode:
            self.model_names.append('Classify')
            self.Classify = define_net(net_type='classify', input_nc=self.opt.input_nc, output_nc=self.opt.input_nc,
                            ngf=self.opt.ngf, nc=self.opt.c_num + self.opt.d_num, e_blocks=self.opt.e_blocks,
                            norm_type=self.opt.norm, gpu_ids=self.gpu_ids)
            self.criterion_classify = nn.CrossEntropyLoss()
            self.G = define_net(net_type=self.opt.G_mode, input_nc=self.opt.input_nc, output_nc=self.opt.input_nc,
                                ngf=self.opt.ngf, nc=self.opt.c_num + self.opt.d_num, e_blocks=self.opt.e_blocks,
                                norm_type=self.opt.norm, gpu_ids=self.gpu_ids)
        else:  # generator  generator_simple
            self.G = define_net(net_type=self.opt.G_mode, input_nc=self.opt.input_nc, output_nc=self.opt.input_nc,
                                ngf=self.opt.ngf, nc=self.opt.c_num + self.opt.d_num, e_blocks=self.opt.e_blocks,
                                norm_type=self.opt.norm, gpu_ids=self.gpu_ids)

        ################### encoder ###########################################
        self.E = None
        if self.opt.mode == 'multimodal':
            self.E = Encoder(input_nc=self.opt.input_nc, output_nc=self.opt.c_num, nef=self.opt.nef, nd=self.opt.d_num,
                             n_blocks=4, norm_type=self.opt.norm)
        if self.opt.isTrain:
            ################### discriminators #####################################
            self.Ds = []
            if self.opt.classify_mode:
                for i in range(self.opt.d_num):
                    self.Ds.append(
                        define_net(net_type=self.opt.D_mode, input_nc=self.opt.output_nc, ndf=self.opt.ndf, block_num=3,
                                   norm_type=self.opt.norm, gpu_ids=self.gpu_ids))
            else:
                for i in range(self.opt.d_num):
                    self.Ds.append(  # discriminator
                        define_net(net_type=self.opt.D_mode, input_nc=self.opt.output_nc, ndf=self.opt.ndf, block_num=3,
                                   norm_type=self.opt.norm, gpu_ids=self.gpu_ids))
            ################### init_weights ########################################
            if self.opt.continue_train:
                if not self.opt.using_save:
                    self.G.load_state_dict(torch.load('{}/G_{}.pth'.format(self.opt.model_dir, self.opt.which_epoch)))
                    if self.E is not None:
                        self.E.load_state_dict(
                            torch.load('{}/E_{}.pth'.format(self.opt.model_dir, self.opt.which_epoch)))
                    for i in range(self.opt.d_num):
                        self.Ds[i].load_state_dict(
                            torch.load('{}/D_{}_{}.pth'.format(self.opt.model_dir, i, self.opt.which_epoch)))
                else:
                    self.G.load_state_dict(
                        torch.load('{}/G_{}save.pth'.format(self.opt.model_dir, self.opt.which_epoch)))
                    for i in range(self.opt.d_num):
                        self.Ds[i].load_state_dict(
                            torch.load('{}/D_{}_{}save.pth'.format(self.opt.model_dir, i, self.opt.which_epoch)))

                # else:
            #     self.G.apply(weights_init(self.opt.init_type))
            #     if self.E is not None:
            #         self.E.apply(weights_init(self.opt.init_type))
            #     for i in range(self.opt.d_num):
            #         self.Ds[i].apply(weights_init(self.opt.init_type))
            ################### use GPU #############################################
            # self.G.cuda()
            # if self.E is not None:
            #     self.E.cuda()
            # for i in range(self.opt.d_num):
            #     self.Ds[i].cuda()
            # if self.opt.classify_mode:
            #     self.Classify.cuda()
            ################### set criterion ########################################
            self.criterionGAN = GANLoss(mse_loss=(self.opt.c_gan_mode == 'lsgan'))
            ################## define optimizers #####################################
            self.define_optimizers()
        else:
            if not self.opt.using_save:
                self.G.load_state_dict(torch.load('{}/G_{}.pth'.format(self.opt.model_dir, self.opt.which_epoch)))
            else:
                self.G.load_state_dict(torch.load('{}/G_{}save.pth'.format(self.opt.model_dir, self.opt.which_epoch)))
            if len(self.gpu_ids) > 0:
                assert (torch.cuda.is_available())
                self.G.to(self.device)
                # self.G = torch.nn.DataParallel(self.G, self.gpu_ids)
            self.G.eval()
            if self.E is not None:
                self.E.load_state_dict(torch.load('{}/E_{}.pth'.format(self.opt.model_dir, self.opt.which_epoch)))
                self.E.cuda()
                self.E.eval()

    def sample_latent_code(self, size):
        c = torch.cuda.FloatTensor(size).normal_()
        return Variable(c)

    def get_domain_code(self, domainLable):
        domainCode = torch.zeros([len(domainLable), self.opt.d_num])
        domainIndex_cache = [[] for i in range(self.opt.d_num)]
        for index in range(len(domainLable)):
            domainCode[index, domainLable[index]] = 1
            domainIndex_cache[domainLable[index]].append(index)
        domainIndex = []
        for index in domainIndex_cache:
            domainIndex.append(Variable(torch.LongTensor(index)).cuda())
        return Variable(domainCode).cuda(), domainIndex

    def define_optimizer(self, Net):
        return optim.Adam(Net.parameters(),
                          lr=self.opt.lr,
                          betas=(0.5, 0.999))

    def define_optimizers(self):
        self.G_opt = self.define_optimizer(self.G)
        self.E_opt = None
        if self.E is not None:
            self.E_opt = self.define_optimizer(self.E)
        self.Ds_opt = []
        for i in range(self.opt.d_num):
            self.Ds_opt.append(self.define_optimizer(self.Ds[i]))
        if self.opt.classify_mode:
            self.Classify_opt=optim.Adam(self.Classify.parameters(),
                       lr=self.opt.lr/2,
                       betas=(0.5, 0.999))

    def update_lr(self, lr):
        for param_group in self.G_opt.param_groups:
            param_group['lr'] = lr
        if self.E_opt is not None:
            for param_group in self.E_opt.param_groups:
                param_group['lr'] = lr
        for i in range(self.opt.d_num):
            for param_group in self.Ds_opt[i].param_groups:
                param_group['lr'] = lr

    def save(self, name):
        torch.save(self.G.state_dict(), '{}/G_{}.pth'.format(self.opt.model_dir, name))
        if self.E_opt is not None:
            torch.save(self.E.state_dict(), '{}/E_{}.pth'.format(self.opt.model_dir, name))
        for i in range(self.opt.d_num):
            torch.save(self.Ds[i].state_dict(), '{}/D_{}_{}.pth'.format(self.opt.model_dir, i, name))

    def prepare_image(self, data):
        img, sourceD, targetD = data
        return Variable(torch.cat(img, 0)).cuda(), torch.cat(sourceD, 0), torch.cat(targetD, 0)

    def translation(self, data):
        input, sourceD, targetD = self.prepare_image(data)
        sourceDC, sourceIndex = self.get_domain_code(sourceD)
        targetDC, targetIndex = self.get_domain_code(targetD)

        images, names = [], []
        # for i in range(self.opt.d_num):
        #     images.append([tensor2im(input.index_select(0,sourceIndex[i])[0].data)])
        #     names.append(['D{}'.format(i)])

        if self.opt.mode == 'multimodal':
            for i in range(self.opt.n_samples):
                c_rand = self.sample_latent_code(torch.Size([input.size(0), self.opt.c_num]))
                targetC = torch.cat([targetDC, c_rand], 1)
                output = self.G(input, targetC)
                for j in range(output.size(0)):
                    images[sourceD[j]].append(tensor2im(output[j].data))
                    names[sourceD[j]].append('{}to{}_{}'.format(sourceD[j], targetD[j], i))
        else:
            output = self.G(input, targetDC)  # [sourceD[i]]  [sourceD[i]]
            for i in range(output.size(0)):
                images.append(tensor2imMy(output[i].data))
                names.append('fake'.format(targetD[i]))

        return images

    def get_current_errors(self):
        dict = []
        for i in range(self.opt.d_num):
            dict += [('D_{}'.format(i), self.errDs[i].item())]
            dict += [('G_{}'.format(i), self.errGs[i].item())]
        dict += [('errCyc', self.errCyc.item())]
        if self.opt.lambda_ide > 0:
            dict += [('errIde', self.errIde.item())]
        if self.E is not None:
            dict += [('errKl', self.errKL.item())]
            dict += [('errCode', self.errCode.item())]
        return OrderedDict(dict)

    def get_current_visuals(self):
        real = make_grid(self.real.data, nrow=self.real.size(0), padding=0)
        fake = make_grid(self.fake.data, nrow=self.real.size(0), padding=0)
        cyc = make_grid(self.cyc.data, nrow=self.real.size(0), padding=0)
        img = [real, fake, cyc]
        name = 'rsal,fake,cyc'
        if self.opt.lambda_ide > 0:
            ide = make_grid(self.ide.data, nrow=self.real.size(0), padding=0)
            img.append(ide)
            name += ',ide'
        img = torch.cat(img, 1)
        return OrderedDict([(name, tensor2im(img))])

    def update_D(self, D, D_opt, real, fake):
        D.zero_grad()
        pred_fake = D(fake.detach())
        pred_real = D(real)
        errD = 0.5 * (self.criterionGAN(pred_fake, False) + self.criterionGAN(pred_real, True))
        errD.backward()
        D_opt.step()
        return errD

    def calculate_G(self, D, fake):
        pred_fake = D(fake)
        errG = self.criterionGAN(pred_fake, True)
        return errG

    def prepare_classify_image(self, data):
        img, sourceD, targetD, label = data
        img = torch.stack([t for t in img], 0)
        # sourceD=torch.stack( [t for t in sourceD],0)
        # targetD = torch.stack([t for t in targetD], 0)
        label = torch.stack([t for t in label], 0)
        return Variable(img).cuda(), torch.cat(sourceD, 0), torch.cat(targetD, 0), label.cuda()  # torch.cat(label, 0)

    def classify_train(self, data):
        self.real, sourceD, targetD, label = self.prepare_classify_image(data)
        targetDC, self.targetIndex = self.get_domain_code(targetD)
        targetC = targetDC
        self.fake = self.G(self.real, targetC)
        outputs = self.Classify(torch.cat([self.fake, self.fake, self.fake], 1))
        self.Classify.zero_grad()
        self.G.zero_grad()
        loss = self.criterion_classify(outputs, label)
        _, preds = torch.max(outputs, 1)
        acc = torch.sum(preds == label)
        loss.backward()
        self.Classify_opt.step()
        self.G_opt.step()
        return acc.cpu().float()

    def update_model(self, data):
        ratio = 0.5
        ### prepare data ###
        self.real, sourceD, targetD = self.prepare_image(data)
        sourceDC, self.sourceIndex = self.get_domain_code(sourceD)
        targetDC, self.targetIndex = self.get_domain_code(targetD)
        sourceC, targetC = sourceDC, targetDC
        ### generate image ###
        if self.E is not None:
            c_enc, mu, logvar = self.E(self.real, sourceDC)
            c_rand = self.sample_latent_code(c_enc.size())
            sourceC = torch.cat([sourceDC, c_enc], 1)
            targetC = torch.cat([targetDC, c_rand], 1)
        self.fake = self.G(self.real, targetC)
        self.cyc = self.G(self.fake, sourceC)
        if self.E is not None:
            _, mu_enc, _ = self.E(self.fake, targetDC)
        if self.opt.lambda_ide > 0:
            self.ide = self.G(self.real, sourceC)

        ### update D ###
        self.set_requires_grad(self.Ds, requires_grad=True)
        self.errDs = []
        for i in range(self.opt.d_num):
            errD = self.update_D(self.Ds[i], self.Ds_opt[i], self.real.index_select(0, self.sourceIndex[i]),
                                 self.fake.index_select(0, self.targetIndex[i]))
            self.errDs.append(errD)

        self.set_requires_grad(self.Ds, requires_grad=False)
        ### update G ###
        self.errGs, self.errKl, self.errCode, errG_total = [], 0, 0, 0
        self.G.zero_grad()
        for i in range(self.opt.d_num):
            errG = self.calculate_G(self.Ds[i], self.fake.index_select(0, self.targetIndex[i]))
            errG_total += errG
            self.errGs.append(errG)
        self.errCyc = (torch.mean(torch.abs(self.cyc - self.real)) * (1 - ratio) +
                       ratio * mssim_loss(self.cyc, self.real)) \
                      * self.opt.lambda_cyc
        errG_total += self.errCyc
        if self.opt.lambda_ide > 0:
            self.errIde = (torch.mean(torch.abs(self.ide - self.real)) * (1 - ratio) +
                           ratio * mssim_loss(self.ide, self.real)) \
                          * self.opt.lambda_ide * self.opt.lambda_cyc
            errG_total += self.errIde
        if self.E is not None:
            self.E.zero_grad()
            self.errKL = KL_loss(mu, logvar) * self.opt.lambda_kl
            errG_total += self.errKL
            errG_total.backward(retain_graph=True)
            self.G_opt.step()
            self.E_opt.step()
            self.G.zero_grad()
            self.E.zero_grad()
            self.errCode = torch.mean(torch.abs(mu_enc - c_rand)) * self.opt.lambda_c
            self.errCode.backward()
            self.G_opt.step()
        else:
            errG_total.backward()
            self.G_opt.step()

    def eval(self):
        self.G.eval()

    def train(self):
        self.G.train(mode=True)

    def set_requires_grad(self, nets, requires_grad=False):
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

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                num_params = 0
                if isinstance(net, list):
                    for sub_net in net:
                        for param in sub_net.parameters():
                            num_params += param.numel()
                else:
                    for param in net.parameters():
                        num_params += param.numel()
                # if verbose:
                #    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')
