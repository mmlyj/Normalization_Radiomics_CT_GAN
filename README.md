# Original project 
https://github.com/Xiaoming-Yu/SingleGAN ,code of this project inherited from SingleGAN
# Normalization
A GAN method for normalizations of CT images
# Train
python train.py --name XXX  --dataroot /XXX   --D_mode discriminator --G_mode  generator_classify  --d_num 4 --matmode  --lambda_cyc  40  --lambda_ide  2 --results_dir /XXX 
# Test
python testAll.py --name GEPhilips2Siemens_ide2  --dataroot XXX --D_mode discriminator --G_mode  generator_classify   --d_num 4 --matmode  --using_save  --time_dir  2019_07_18_22_52_24  --how_many XXX
# paper
normalization of multicenter ct radiomics by a generative adversarial network method
