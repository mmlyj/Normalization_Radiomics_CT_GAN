import os
from options.test_options import TestOptions
from data.testloader import CreateDataLoaderMy
from util.visualizer import save_images,save_mat
from itertools import islice
from models.single_gan import SingleGAN
from util import html, util
from  util.util import  mkdir
from glob import glob
def main():
    opt = TestOptions().parse()
    opt.no_flip = True
    opt.batchSize = 4
    data_loader = CreateDataLoaderMy(opt)

    opt.matmode=True
    model = SingleGAN()
    for epoch in range(114,115):
        print(' model_dir is %s ' % os.path.isdir(opt.model_dir))
      #  print(opt.model_dir+os.path.sep+"G_save.pth")
       # print(' %d epoch is dealing' % len(glob("{}"+os.path.sep+"G_{}save.pth".format(opt.model_dir,epoch))))
        if  os.path.isfile(opt.model_dir+os.path.sep+"G"+str(epoch)+"_save.pth"):
            print('skip %d epoch' % epoch)
            continue
        print(' %d epoch is dealing' % epoch)
        opt.which_epoch=epoch
        model.initialize(opt)
        web_dir = os.path.join(opt.results_dir, opt.name, 'epoch_%s' %( opt.which_epoch))
        mkdir(web_dir)
       # web_dir = os.path.join(opt.results_dir, 'test')
        webpage = html.HTML(web_dir, 'task {}'.format(opt.name))
        for i, data in enumerate(islice(data_loader, opt.how_many)):
           # print('dataroot%s' %(opt.dataroot))
            print('process input image %3.3d/%3.3d' % (i, opt.how_many))
            all_images = model.translation(data[0:3])
            img_path = data[3]
            if not opt.matmode:
                save_images(web_dir, all_images, img_path)
            else:
                save_mat(web_dir,all_images,img_path)
        util.run_radiomics_test(opt.name, opt.which_epoch, web_dir,True,opt.dataroot)
    #webpage.save()

if __name__ == '__main__':
    main()