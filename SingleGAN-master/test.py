import os
from options.test_options import TestOptions
from data.testloader import CreateDataLoaderMy
from util.visualizer import save_images,save_mat
from itertools import islice
from models.single_gan import SingleGAN
from util import html, util
from  util.util import  mkdir
def main():    
    opt = TestOptions().parse()
    opt.no_flip = True
    #opt.batchSize = 4
    data_loader = CreateDataLoaderMy(opt)
    opt.matmode=True
    model = SingleGAN()
    model.initialize(opt)
    web_dir = os.path.join(opt.results_dir, opt.name, 'epoch_%s' %( opt.which_epoch))
    mkdir(web_dir)
  #  web_dir = os.path.join(opt.results_dir, 'test')
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
   # util.run_radiomics_test(opt.name, opt.which_epoch, web_dir,True,opt.dataroot)
    #webpage.save()

if __name__ == '__main__':
    main()