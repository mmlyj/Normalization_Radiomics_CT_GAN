import os
import random
from glob import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import scipy.io as scio
import torch

def CreateDataLoaderMy(opt):

    if opt.mode == 'one2many':
        sourceD = [0 ]#for i in range(opt.d_num - 1)] + [0 for i in range(1, opt.d_num)]
        targetD = [0 ]#for i in range(1, opt.d_num)] + [0 for i in range(opt.d_num - 1)]
    elif opt.mode == 'many2many':
        sourceD = [i for i in range(opt.d_num) for j in range(opt.d_num)]
        targetD = [j for i in range(opt.d_num) for j in range(opt.d_num)]
    else:
        raise ('mode:{} does not exist'.format(opt.mode))
    batchSize = 1
    dataset = UnPairedDataset(opt.dataroot,
                              sourceD=sourceD,
                              targetD=targetD,
                              matmode=opt.matmode,is3d=opt.is3d
                              )
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batchSize,
                             shuffle=opt.isTrain,
                             drop_last=False,
                             num_workers=opt.nThreads)

    return data_loader


class UnPairedDataset(Dataset):
    def __init__(self, image_path, sourceD=[0, 1], targetD=[1, 0],matmode=False,is3d=False):
        self.image_path = image_path
        self.sourceD = sourceD
        self.targetD = targetD
        self.mat_mode=matmode
        self.mat_mode=True
        self.is3d=is3d
        print('Start preprocessing dataset..!')
        self.preprocess()
        print('Finished preprocessing dataset..!')
        if not self.mat_mode:
            trs = [transforms.Grayscale(1), transforms.CenterCrop([512, 512])]
            self.transform = transforms.Compose(trs)
            self.norm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])])
        else:
            self.transform = transforms.Compose([
                transforms.Normalize([0.5],
                                     [0.5])])
        self.num_data = max(self.num)
        print('Finished preprocessing dataset..%d!', self.num_data)
    def preprocess(self):
        self.filenames = []
        self.num = []
        print('image path:%s' % self.image_path)
        if not self.mat_mode :
            filenames = glob("{}/*.png".format(self.image_path))
        else:
            filenames = glob("{}/*.mat".format(self.image_path))
        filenames.sort()
        self.filenames.append(filenames)
        self.num.append(len(filenames))
        print('image path:%d' % len(filenames))
    def __getitem__(self, index):
        imgs = []

        #for d in self.sourceD:
        d=0
        index_d = index if index < self.num[d] else random.randint(0,self.num[d]-1)

        if not self.mat_mode:
            img = Image.open(self.filenames[d][index_d]).convert('RGB')
            img = self.transform(img)
            img = self.norm(img)
            imgs.append(img)
        else:
            data = scio.loadmat(self.filenames[d][index_d])
            img = data['img']
            img = torch.from_numpy(img)
            if not self.is3d:
                img = img.unsqueeze(0).float()
                img = self.transform(img.float())
            else:
                img[img<=-200]=-200
                img[img >= 300] = 300
                img=(img.float()+200)/500
                img=(img-0.5)/0.5
                img=img.permute(2, 0,1)
            imgs.append(img)

        return imgs, self.sourceD, self.targetD,self.filenames[d][index_d]

    def __len__(self):
        return self.num_data
