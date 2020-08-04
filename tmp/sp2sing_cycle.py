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
speech_dataset = AudioNpyLoader(hp.speech_files)
singing_dataset = AudioNpyLoader(hp.singing_files)

sp_iterator_tr = DataLoader(
        speech_dataset,
        batch_size=4,
        num_workers=2,
        shuffle=True,
        drop_last=True,
        pin_memory=True, collate_fn = AudioCollate())

inf_iterator_tr_speech = make_inf_iterator(sp_iterator_tr)

si_iterator_tr = DataLoader(
        singing_dataset,
        batch_size=4,
        num_workers=2,
        shuffle=True,
        drop_last=True,
        pin_memory=True, collate_fn = AudioCollate())

inf_iterator_tr_sing = make_inf_iterator(si_iterator_tr)

##################################################################
# BEGAN parameters
if hp.loss == "BEGAN":
    gamma = 1.0
    lambda_k = 0.01
    init_k = 0.0
    recorder = BEGANRecorder(lambda_k, init_k, gamma)
    k = recorder.k.item()
criterion = nn.L1Loss( reduce = False)

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
    
    
    speech = next(inf_iterator_tr_speech).cuda()
    singing = next(inf_iterator_tr_sing).cuda()
    speech_2x= F.interpolate(speech, scale_factor=2, mode='nearest')
    #mel = (mel-mean)/std
    singing = singing[:,:,:min(speech_2x.size(2), singing.size(2))]
    speech_2x = speech_2x[:,:,:min(speech_2x.size(2), singing.size(2))]
    

    singing = F.pad(singing,(0,(singing.size(2)//64+1)*64 - singing.size(2)), 'reflect')
    speech_2x = F.pad(speech_2x,(0,(speech_2x.size(2)//64+1)*64 - speech_2x.size(2)), 'reflect')
    fake_singing = speech_2x
    for (i,block) in enumerate(m):
        if i == 0:
            fake_singing = block(fake_singing)
        else:
            fake_speech = block(fake_singing)
    if hp.loss == 'BEGAN':
        loss_gan, loss_dis, real_dloss, fake_dloss = BEGANLoss(dis_high, singing, fake_singing, k)
        loss_cycle = criterion(speech_2x, fake_speech).mean()

        OptimStep([(m, opt, loss_gan + 0.2 * loss_cycle, True),
            (dis_high, opt_dis, loss_dis, False)], 3)
        
        k, convergence = recorder(real_dloss, fake_dloss, update_k=True)
    
    if iteration % 5 == 0:
        if hp.loss == "BEGAN":
            logger.log_training(iteration = iteration, loss_gan = loss_gan, 
            loss_dis = loss_dis, loss_cycle = loss_cycle, k = k, convergence = convergence)

    if (iteration % 50 == 0):

        save_checkpoint(m, opt, iteration, f'checkpoint/{args.checkpoint_path}/gen')
        save_checkpoint(dis_high, opt_dis, iteration, f'checkpoint/{args.checkpoint_path}/dis')

        
        idx = random.randint(0, fake_singing.size(0) - 1)

        #mel = (mel * std) +mean
        #z = (z * std) + mean
        real_audio = melblock.inverse(singing).detach().cpu().numpy()
        fake_audio = melblock.inverse(fake_singing).detach().cpu().numpy()
        real_speech_audio = melblock.inverse(speech).detach().cpu().numpy()
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
            real_sing = ("image", plot_spectrogram_to_numpy(), singing[idx]),
            fake_sing = ("image", plot_spectrogram_to_numpy(), fake_singing[idx]),
            real_speech = ("image", plot_spectrogram_to_numpy(), speech[idx]),

            real_sing_audio = ("audio", 22050, real_audio[idx]),
            fake_sing_audio = ("audio", 22050, fake_audio[idx]),
            real_speech_audio = ("audio", 22050, real_speech_audio[idx])
        )
    logger.close()
    iteration += 1