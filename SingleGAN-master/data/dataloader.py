import torch
import os
import random
from glob import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

def CreateDataLoader(opt):
    if opt.mode == 'base' or opt.mode == 'multimodal':
        sourceD, targetD = [0,1], [1,0]
    elif opt.mode == 'one2many':
        sourceD = [0 for i in range(opt.d_num-1)] + [i for i in range(1, opt.d_num)]
        targetD = [i for i in range(1, opt.d_num)] + [0 for i in range(opt.d_num-1)]
    elif opt.mode == 'many2many':
        sourceD = [i for i in range(opt.d_num) for j in range(opt.d_num)]
        targetD = [j for i in range(opt.d_num) for j in range(opt.d_num)]
    else:
        raise('mode:{} does not exist'.format(opt.mode))
    
    dataset = UnPairedDataset(opt.dataroot,
                            opt.loadSize,
                            opt.fineSize,
                            opt.is_flip>0,
                            opt.isTrain,
                            sourceD=sourceD,
                            targetD=targetD
                            )
    data_loader = DataLoader(dataset=dataset,
                             batch_size=opt.batchSize,
                             shuffle=opt.isTrain,
                             drop_last=True,
                             num_workers=opt.nThreads)
                             
    return data_loader

class UnPairedDataset(Dataset):
    def __init__(self, image_path, loadSize, fineSize, isFlip, isTrain, sourceD=[0,1], targetD=[1,0]):
        self.image_path = image_path
        self.isTrain = isTrain
        self.fineSize = fineSize
        self.sourceD = sourceD
        self.targetD = targetD
        print ('Start preprocessing dataset..!')
        random.seed(1234)
        self.mutiAligned = True
        self.max_position = 13
        p = np.zeros([self.max_position - 1], dtype=np.float16)
        p[0:(self.max_position - 1)] = 1 / (self.max_position-1)
        # p[12:self.max_position - 1] = 0.1 / (self.max_position - 1 - 12)
        self.p=p
        self.random_num=np.linspace(0,self.max_position-2,self.max_position-1,dtype=np.int)
        self.position=0
        if self.mutiAligned :
            self.preprocessMuti()
        else:
            self.preprocess()
        print ('Finished preprocessing dataset..!')
        if isTrain:#transforms.CenterCrop([512, 512])#[400, 400]
            trs = [ transforms.Grayscale(1),transforms.Pad(200), transforms.CenterCrop([512, 512]), transforms.RandomCrop(fineSize)]
        else:
            trs = [ transforms.Grayscale(1),transforms.Resize(loadSize, interpolation=Image.ANTIALIAS), transforms.CenterCrop(fineSize)]
        if isFlip:
            trs.append(transforms.RandomHorizontalFlip())
        self.transform = transforms.Compose(trs)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])])
        self.num_data = max(self.num)
        
    def preprocess(self):
        dirs = os.listdir(self.image_path)
        trainDirs = [dir for dir in dirs if 'train' in dir]
        testDirs = [dir for dir in dirs if 'test' in dir]
        assert len(trainDirs) >=  max(self.sourceD)+1 and len(trainDirs) >=  max(self.targetD)+1
        trainDirs.sort()
        testDirs.sort()
        self.filenames_0 = []
        self.filenames_1 = []
        self.num_0 = []
        self.num_1 = []
        self.num=[]


        if self.isTrain:
            for dir in trainDirs:
                filenames_0 = glob("{}/{}/*_0_.png".format(self.image_path,dir))
                random.shuffle(filenames_0)
                self.filenames_0.append(filenames_0)
                self.num_0.append(len(filenames_0))
                filenames_1 = glob("{}/{}/*_1_.png".format(self.image_path,dir))
                random.shuffle(filenames_1)
                self.filenames_1.append(filenames_1)
                self.num_1.append(len(filenames_1))
                self.num.append(len(filenames_1)+len(filenames_0))
        else:
            for dir in testDirs:
                filenames_0 = glob("{}/{}/*.png".format(self.image_path,dir))
                filenames_0.sort()
                self.filenames_0.append(filenames_0)
                self.num.append(len(filenames_0))
    def preprocessMuti(self):
        dirs = os.listdir(self.image_path)
        trainDirs = [dir for dir in dirs if 'train' in dir]
        testDirs = [dir for dir in dirs if 'test' in dir]
        assert len(trainDirs) >=  max(self.sourceD)+1 and len(trainDirs) >=  max(self.targetD)+1
        trainDirs.sort()
        testDirs.sort()
        self.filenames_0 = []
        self.filenames_1 = []
        self.num_0 = []
        self.num_1 = []
        self.num=[]
        self.filenames=[]
        print('target:%s' % trainDirs[0])
        if self.isTrain:
                dir = trainDirs[0]
                filenames = glob("{}/{}/*.png".format(self.image_path,dir))
                random.shuffle(filenames)
                self.filenames=filenames
                self.num.append(len(filenames))
        if self.isTrain:
            for dir in trainDirs:
                print('target:%s' % dir)
                temp_0=[]
                temp_1=[]
                temp_file_0 =[]
                temp_file_1 = []
                for flag in range(1,self.max_position):
                            filenames_0 = glob("{}/{}/*flag{}_0_.png".format(
                                self.image_path,dir,str(flag)))
                            if filenames_0!=[]:
                                random.shuffle(filenames_0)
                                temp_file_0.append(filenames_0)
                            else:
                                temp_file_0.append([])
                            temp_0.append(len(filenames_0))

                            filenames_1 = glob("{}/{}/*flag{}_1_.png".format(
                                self.image_path, dir, str(flag)))
                            if filenames_1 != []:
                                random.shuffle(filenames_1)
                                temp_file_1.append(filenames_1)

                            else:
                                temp_file_1.append([])
                            temp_1.append(len(filenames_1))
                            self.num.append(len(filenames_1)+len(filenames_0))
                self.filenames_0.append(temp_file_0)
                self.filenames_1.append(temp_file_1)
                self.num_0.append(temp_0)
                self.num_1.append(temp_1)
        else:
            for dir in testDirs:
                filenames_0 = glob("{}/{}/*.png".format(self.image_path,dir))
                filenames_0.sort()
                self.filenames_0.append(filenames_0)
                self.num.append(len(filenames_0))
 
    def __getitem__(self, index):
        imgs = []
        if not self.mutiAligned:
            flag = random.randint(0,1)
            for d in self.sourceD:
                if flag==0:
                    index_d = index if index < self.num_0[d] else random.randint(0, self.num_0[d] - 1)
                    img = Image.open(self.filenames_0[d][index_d]).convert('RGB')
                else:
                    index_d = index if index < self.num_1[d] else random.randint(0, self.num_1[d] - 1)
                    img = Image.open(self.filenames_1[d][index_d]).convert('RGB')
                img = self.transform(img)
                img = self.norm(img)
                imgs.append(img)
            return imgs, self.sourceD, self.targetD
        else:
            #random.randint(0, 1)
          #  flag_position=self.position%(self.max_position-1)
            #flag_position = np.random.choice(self.random_num, p=self.p.ravel())
            file_names=self.filenames[index]
            flag_suffix = int(file_names[-6])
            flag_position=int(file_names.split('/')[-1].split('_')[-3][4:])-1
          #  assert flag_position is not None
            if flag_position > self.max_position-2:
                flag_position=np.random.choice(self.random_num, p=self.p.ravel())
                #flag_position=random.randint(0,self.max_position-2)
            #flag_position =random.randint(0, self.max_position-2)
            for d in self.sourceD:
                index_d,position,returnFlag=self.schedule(flag_suffix,index, d, flag_position)
                #print('index_d:%d,position:%d,returnFlag:%d'%(index_d,position,returnFlag))
                if returnFlag == 0:
                    img = Image.open(self.filenames_0[d][position][index_d]).convert('RGB')
                else:
                    img = Image.open(self.filenames_1[d][position][index_d]).convert('RGB')
                img = self.transform(img)
                img = self.norm(img)
                imgs.append(img)
           # print('images length %d'% len(imgs))
            return imgs, self.sourceD, self.targetD
    def __len__(self):
        return self.num_data
    def mapping(self,position):
        if position<=1:
            return random.randint(0,1)
        elif position<=3:
            return random.randint(2, 3)
        elif position<=5:
            return random.randint(4, 5)
        elif position <=7 :
            return random.randint(6, 7)
        elif position <=9:
            return random.randint(8, 9)
        else:
            return random.randint(10, self.max_position-2)
        # else:
        #     return random.randint(18,self.max_position-2)
    def get_right_position(self,numList,index, d, position):
        count = 0
       # position = self.mapping(position)
        while numList[d][position] == 0 and count<=20:
            position = self.mapping(position)
            count += 1
        if numList[d][position]!=0:
            index_d = index if index < numList[d][position] else index % numList[d][position]#random.randint(0, numList[d][position] - 1)
        else:
            index_d=-1
        return index_d,position
    def schedule(self,flag_suffix,index, d, position):
        index_d=-1
        returnFlag=flag_suffix
        position_new=position
        if flag_suffix==0:
            while  index_d==-1:
                index_d,position_new=self.get_right_position(numList=self.num_0,index=index, d=d, position=position)
                returnFlag=0
                if index_d==-1:
                    index_d ,position_new= self.get_right_position(numList=self.num_1, index=index, d=d, position=position)
                    returnFlag = 1
                    if index_d ==-1:
                        position = self.mapping(position - 1)
        else:
            while index_d == -1:
                index_d ,position_new= self.get_right_position(numList=self.num_1, index=index, d=d, position=position)
                returnFlag = 1
                if index_d == -1:
                    index_d ,position_new= self.get_right_position(numList=self.num_0, index=index, d=d, position=position)
                    returnFlag = 0
                    if index_d == -1:
                        position =self.mapping(position-1)

        return index_d , position_new, returnFlag