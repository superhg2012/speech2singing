import sys
import os
import argparse
import warnings
import yaml
import random

warnings.filterwarnings('ignore')

import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn import functional as F

sys.path.append('model/')

sys.path.append('utils/')
from hparams import *
from save_and_load import save_checkpoint, load_checkpoint
from process_yaml_model import YamlModelProcesser
from optim_step import OptimStep
from dataloader import AudioNpyLoader, AudioCollate, make_inf_iterator
from melspec import MelSpectrogram

sys.path.append('logger')
from logger import Logger
from logger_utils import prepare_directories_and_logger
from plotting_utils import  plot_spectrogram_to_numpy

sys.path.append('loss')
from began_loss import BEGANRecorder, BEGANLoss


parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output_directory', type=str,
                    help='directory to save checkpoints')
parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                    required=False, help='checkpoint path')
parser.add_argument('--rank', type=str, default="0",
                    required=False, help='rank of current gpu')
parser.add_argument('--load_checkpoint', type=bool, default=False,
                    required=False)
parser.add_argument('--hp_config', type=str,
                    required=True, help='hparams configs')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.rank

## hyperparamerter
hp = create_hparams(f"hp_config/{args.hp_config}")

## create logger
logger = prepare_directories_and_logger(Logger, output_directory = f'output/{args.output_directory}')

'''
## DataLoader Part
The directory of the hp.training_files like this:

    - hp.training_files
        - aaa.mp3
        - aaa1.mp3
        - aa2.mp3
        .
        .
        .
    every mp3 is clip to the same length 
    (hp.training_files: ../clips.5s/ )
'''
dataset = AudioNpyLoader(hp.training_files)

data_dir = f'/home/ericwudayi/nas189/homes/kevinco27/dataset/LJSpeech-1.1/clips.5s.mel'
mean_fp = os.path.join(data_dir, f'mean.mel.melgan.npy')
std_fp = os.path.join(data_dir, f'std.mel.melgan.npy')

mean = torch.from_numpy(np.load(mean_fp)).float().cuda().view(1, 80, 1)
std = torch.from_numpy(np.load(std_fp)).float().cuda().view(1, 80, 1)

iterator_tr = DataLoader(
        dataset,
        batch_size=4,
        num_workers=2,
        shuffle=True,
        drop_last=True,
        pin_memory=True)

inf_iterator_tr_speech = make_inf_iterator(iterator_tr)

##################################################################
# BEGAN parameters
if hp.loss == "BEGAN":
    gamma = 1.0
    lambda_k = 0.01
    init_k = 0.0
    recorder = BEGANRecorder(lambda_k, init_k, gamma)
    k = recorder.k.item()

###################################################################
"""
    Model Architecture from General Model

    All Model is wrap by GeneralModel(nn.modules), but you can ignore
    it. It is still in experiments step. The main block here is just 
        Generator : NetG
        Discriminator : NetD
    They are defined in model/layers.py
"""
###################################################################
ymp = YamlModelProcesser()
netG = []
filelist = os.listdir(f'model_config/{hp.config_gen}')
filelist = sorted(filelist, key = lambda x : int(x[0]))
for f in filelist:
    netG += [ymp.construct_model(f"model_config/{hp.config_gen}/{f}")]
print (netG)
m = nn.ModuleList(netG)
m = m.cuda()
opt = optim.Adam(m.parameters(),lr=1e-4)


dis_high = ymp.construct_model(f"model_config/{hp.config_dis}/1.yaml")
dis_high = dis_high.cuda()
opt_dis = optim.Adam(dis_high.parameters(),lr=1e-4)

iteration = 0
if args.load_checkpoint==True:
    m, opt, iteration = load_checkpoint(f'checkpoint/{args.checkpoint_path}/gen', m, opt)       
    dis_high, opt_dis, iteration = load_checkpoint(f'checkpoint/{args.checkpoint_path}/dis', dis_high, opt_dis)

'''
###########################################################
    In general, we preprocess data to npy, and put them in 
    specific folder. Dataloader load npy file. 

    But in this example, I show that how to transfrom audio
    into stft, melspectrogram by torch.nn.module (MelSpectrogram).
###########################################################
'''

melblock = MelSpectrogram(hp).cuda()
while True:
    
    
    mel = next(inf_iterator_tr_speech).cuda()
    
    #mel = (mel-mean)/std
    mel = F.pad(mel,(0,(mel.size(2)//64+1)*64 - mel.size(2)), 'reflect')
    
    bs, n_mel_channels, time = mel.size()
    z = torch.zeros((bs, hp.z_dim, int(time/64))).normal_(0, 1).float().cuda()

    for block in m:
        z = block(z)
    
    if hp.loss == 'BEGAN':
        loss_gan, loss_dis, real_dloss, fake_dloss = BEGANLoss(dis_high, mel, z, k)
        
        OptimStep([(m, opt, loss_gan, True),
            (dis_high, opt_dis, loss_dis, False)], 3)
        
        k, convergence = recorder(real_dloss, fake_dloss, update_k=True)
    
    if iteration % 5 == 0:
        if hp.loss == "BEGAN":
            logger.log_training(iteration = iteration, loss_gan = loss_gan, 
            loss_dis = loss_dis, k = k, convergence = convergence)

    if (iteration % 50 == 0):

        save_checkpoint(m, opt, iteration, f'checkpoint/{args.checkpoint_path}/gen')
        save_checkpoint(dis_high, opt_dis, iteration, f'checkpoint/{args.checkpoint_path}/dis')

        
        idx = random.randint(0, z.size(0) - 1)
        mel = mel
        z = z

        #mel = (mel * std) +mean
        #z = (z * std) + mean
        real_audio = melblock.inverse(mel).detach().cpu().numpy()
        fake_audio = melblock.inverse(z).detach().cpu().numpy()

        #mel = (mel -mean)/ std
        #z = (z - mean ) / std
        """
        logger work like this:
            logger only accept image, audio ,scalars type.
            and the type of them is :
            
            scalars : int
            image : tensor
            audio : ndarray
        
        More details can see logger/logger.py
        """
        logger.log_validation(
            iteration = iteration,
            real_spec = ("image", plot_spectrogram_to_numpy(), mel[idx]),
            fake_spec = ("image", plot_spectrogram_to_numpy(), z[idx]),

            real_audio = ("audio", 22050, real_audio[idx]),
            fake_audio = ("audio", 22050, fake_audio[idx]),
        )
    logger.close()
    iteration += 1