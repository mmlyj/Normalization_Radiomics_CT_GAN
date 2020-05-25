import time
from options.train_options import TrainOptions
from data.dataloader import CreateDataLoader
from data.testloader import CreateDataLoaderMy
from util.visualizer import Visualizer
from models.single_gan import SingleGAN
import os
from util.html import  html
from itertools import islice
from util.visualizer import save_images,save_mat
from util import  util
from  util.util import  mkdir
def main():
    opt = TrainOptions().parse()
    data_loader = CreateDataLoader(opt)
    dataset_size = len(data_loader)
    visualizer = Visualizer(opt)
    test_loader = CreateDataLoaderMy(opt)
    test_size=len(test_loader)
    print('The number of training images = %d' % dataset_size)
    print('The number of test_size  = %d' % test_size)
    #webpage = html.HTML(web_dir, 'task {}'.format(opt.name))

    model = SingleGAN()
    model.initialize(opt)
    model.print_networks(False)
    print('opt.batchSize = %d' %  opt.batchSize)
    total_steps = 0
    start_test=0
    save_count=112
    lr = opt.lr
    opt.batchSize=1
    for epoch in range(opt.epoch_count, 200):
        epoch_start_time = time.time()
        save_result = True
        epoch_iter=0
        for i, data in enumerate(data_loader):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize  #total_steps - dataset_size * (epoch - 1)
            model.update_model(data)
            start_test+=opt.batchSize
            if save_result or total_steps % opt.display_freq == 0:
                save_result = save_result or total_steps % opt.update_html_freq == 0
                print('mode:{} dataset:{}'.format(opt.mode,opt.name))
                visualizer.display_current_results(model.get_current_visuals(), epoch, ncols=1, save_result=save_result)
                save_result = False
            
            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)
                    
            if  start_test>=opt.save_latest_freq:
                start_test=0
                save_count=save_count+1
                print('saving the latest model (epoch %d, total_steps %d)' %(epoch, total_steps))
                model.save(str(save_count)+'save')
                model.save('latest')
                model.eval()
                model.set_requires_grad(model.G, False)
                web_dir = os.path.join(opt.results_dir, opt.name, 'epoch_%s' % (epoch))
                mkdir(web_dir)
                for j, data_test in enumerate(test_loader):
                    print('process input image %3.3d/%3.3d' % (j, test_size))
                    all_images = model.translation(data_test[0:3])
                    img_path = data_test[3]
                    if not opt.matmode:
                        save_images(web_dir, all_images, img_path )
                    else:
                        save_mat(web_dir, all_images, img_path)
                util.run_radiomics_test(opt.name, str(save_count)+'save', web_dir)
                model.train()
                model.set_requires_grad(model.G,True)
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %(epoch, total_steps))
            model.save(epoch)
            
        if epoch >= opt.niter:
            lr -= epoch* opt.lr / opt.niter_decay
            model.update_lr(lr)
        
if __name__ == '__main__':
    main()