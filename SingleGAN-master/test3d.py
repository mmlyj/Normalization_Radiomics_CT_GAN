import os
from options.test_options import TestOptions
from data.test3dloader import CreateDataLoaderMy
from util.visualizer import save_mat
from itertools import islice
from models.single_gan import SingleGAN
#from util import html, util
from  util.util import  mkdir
from glob import glob
import torch
import math
def main():
    opt = TestOptions().parse()
    #opt.dataroot='/mnt/lzh/3dtest'
    opt.no_flip = True
    opt.batchSize = 1
    data_loader = CreateDataLoaderMy(opt)
    opt.matmode=True
    model = SingleGAN()
    for epoch in range(104,170):
        if len(glob("{}/G_{}save.pth".format(opt.model_dir,epoch)))==0:
            print('skip %d epoch' % epoch)
            continue
        print(' %d epoch is dealing' % epoch)
        opt.which_epoch=epoch
        model.initialize(opt)
        web_dir = os.path.join(opt.results_dir, opt.name, 'epoch_%s' %( opt.which_epoch))
        mkdir(web_dir)
        #webpage = html.HTML(web_dir, 'task {}'.format(opt.name))
        for i, data in enumerate(islice(data_loader, len(data_loader))):
            print('process  patients %3.3d/%3.3d' % (i, len(data_loader)))
            data_3d_raw = data[0][0]
            data_3d_raw=data_3d_raw.permute(1,0,2,3)
            batch_size = 6
            dim=0
            num= math.ceil(data_3d_raw.shape[dim]/batch_size)
            data_3d_img=torch.split(data_3d_raw, batch_size, dim=dim)

            save_3d_img=torch.ones_like(data_3d_raw, dtype=torch.float)
            for index in range(0,num):
                end=min((index+1)*batch_size,data_3d_raw.shape[dim])
                print('process  image th%3.3d/%3.3d' % (end, data_3d_raw.shape[dim]))
                tempSize=end-(index*batch_size)
                source = torch.zeros([tempSize]).long()
                target = torch.zeros([tempSize]).long()
                data_3d_list = [data_3d_img[index], source, target]
                # print(index*batch_size)
                # print(end)
                # print('dim size%d:' % data_3d_raw.shape[dim])
                save_3d_img[(index*batch_size):end,:,:,:] = model.translation3d(data_3d_list)
            img_path = data[3]
            save_3d_img=[save_3d_img.squeeze(1).permute(1,2,0).float().numpy()]
            save_mat(web_dir,save_3d_img,img_path,True)
        #util.run_radiomics_test(opt.name, opt.which_epoch, web_dir,True,opt.dataroot)
    #webpage.save()

if __name__ == '__main__':
    main()