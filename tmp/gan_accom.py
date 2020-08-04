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
from torch.nn import functional as F
from torch.utils.data import DataLoader

sys.path.append('model/')

sys.path.append('utils/')
from hparams import *
from save_and_load import save_checkpoint, load_checkpoint
from process_yaml_model import YamlModelProcesser
from optim_step import OptimStep
from dataloader import AudioLoader, AudioNpyLoader, AudioCollate, make_inf_iterator
from melspec import MelSpectrogram
from stft import STFT

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
vocal_dataset = AudioNpyLoader(hp.vocal_files)
linear_mixture_dataset = AudioNpyLoader(hp.linear_mixture_files)
accom_dataset = AudioNpyLoader(hp.accom_files)

vocal_iterator_tr = DataLoader(
        vocal_dataset,
        batch_size=4,
        num_workers=2,
        shuffle=True,
        drop_last=True,
        pin_memory=True, collate_fn = AudioCollate())

inf_iterator_voc_speech = make_inf_iterator(vocal_iterator_tr)


linear_iterator_tr = DataLoader(
        linear_mixture_dataset,
        batch_size=4,
        num_workers=2,
        shuffle=True,
        drop_last=True,
        pin_memory=True, collate_fn = AudioCollate())

inf_iterator_lin_speech = make_inf_iterator(linear_iterator_tr)

accom_iterator_tr = DataLoader(
        accom_dataset,
        batch_size=4,
        num_workers=2,
        shuffle=True,
        drop_last=True,
        pin_memory=True, collate_fn = AudioCollate())

inf_iterator_accom_speech = make_inf_iterator(accom_iterator_tr)
##################################################################
# BEGAN parameters
if hp.loss == "BEGAN":
    gamma = 1.0
    lambda_k = 0.01
    init_k = 0.0
    recorder = BEGANRecorder(lambda_k, init_k, gamma)
    k = recorder.k.item()

    recorder2 = BEGANRecorder(lambda_k, init_k, gamma)
    k2 = recorder2.k.item()
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


dis_accom = ymp.construct_model(f"model_config/{hp.config_dis_accom}/1.yaml")
dis_accom = dis_accom.cuda()
opt_accom = optim.Adam(dis_accom.parameters(),lr=1e-4)

iteration = 0
if args.load_checkpoint==True:
    m, opt, iteration = load_checkpoint(f'checkpoint/{args.checkpoint_path}/gen', m, opt)       
    dis_high, opt_dis, iteration = load_checkpoint(f'checkpoint/{args.checkpoint_path}/dis', dis_high, opt_dis)
    dis_accom, opt_accom, iteration = load_checkpoint(f'checkpoint/{args.checkpoint_path}/dis_accom', dis_accom, opt_accom)

'''
###########################################################
    In general, we preprocess data to npy, and put them in 
    specific folder. Dataloader load npy file. 

    But in this example, I show that how to transfrom audio
    into stft, melspectrogram by torch.nn.module (MelSpectrogram).
###########################################################
'''
stft_fn = STFT(hp.filter_length, hp.hop_length, hp.win_length).cuda()

while True:
    
    
    voc = next(inf_iterator_voc_speech).cuda()
    #voc = stft_fn.transform_mag(voc)


    linear = next(inf_iterator_lin_speech).cuda()
    #linear = stft_fn.transform_mag(linear)
    accom = next(inf_iterator_accom_speech).cuda()

    voc = voc[..., :voc.size(2)//16 * 16]
    linear = linear[..., :linear.size(2)//16 * 16]
    accom = accom[..., : accom.size(2)//16 * 16]

    fake_accom = voc

    print ("fake_accom size:", fake_accom.size())
    for block in m:
        fake_accom = block(fake_accom)

    fake_linear = (10**fake_accom + 10**voc)/6
    fake_linear = torch.log10(torch.clamp(fake_linear, min=1e-5))

    if hp.loss == 'BEGAN':
        loss_gan, loss_dis, real_dloss, fake_dloss = BEGANLoss(dis_high, linear, fake_linear, k)
        loss_accom, loss_dis_accom, real_acloss, fake_acloss = BEGANLoss(dis_accom, accom, fake_accom, k2)

        OptimStep([(m, opt, loss_gan + loss_accom, True),
            (dis_accom, opt_accom, loss_dis_accom, True),
            (dis_high, opt_dis, loss_dis, False)], 3)
        
        k, convergence = recorder(real_dloss, fake_dloss, update_k=True)
        k2, convergence2 = recorder(real_acloss, fake_acloss, update_k=True)
    if iteration % 5 == 0:
        if hp.loss == "BEGAN":
            logger.log_training(iteration = iteration, loss_gan = loss_gan, 
            loss_dis = loss_dis, k = k, logg_accom = loss_accom, 
            loss_dis_accom = loss_dis_accom, k2 = k2)

    if (iteration % 50 == 0):

        save_checkpoint(m, opt, iteration, f'checkpoint/{args.checkpoint_path}/gen')
        save_checkpoint(dis_high, opt_dis, iteration, f'checkpoint/{args.checkpoint_path}/dis')
        save_checkpoint(dis_accom, opt_accom, iteration, f'checkpoint/{args.checkpoint_path}/dis_accom')

        
        idx = random.randint(0, linear.size(0) - 1)
        
        linear = linear[idx].unsqueeze(0)
        fake_linear = fake_linear[idx].unsqueeze(0)
        voc = voc[idx].unsqueeze(0)
        accom = accom[idx].unsqueeze(0)

        voc_audio = stft_fn.inverse_mag(voc).detach().cpu().numpy()
        accom_audio = stft_fn.inverse_mag(accom).detach().cpu().numpy()
        real_audio = stft_fn.inverse_mag(linear).detach().cpu().numpy()
        fake_audio = stft_fn.inverse_mag(fake_linear).detach().cpu().numpy()
        
        linear = linear.squeeze()
        fake_linear = fake_linear.squeeze()
        voc = voc.squeeze()
        accom = accom.squeeze()
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
            real_spec = ("image", plot_spectrogram_to_numpy(), linear),
            fake_spec = ("image", plot_spectrogram_to_numpy(), fake_linear),
            voc_spec = ("image", plot_spectrogram_to_numpy(), voc),
            accom_spec = ("image", plot_spectrogram_to_numpy(), accom),

            voc_audio = ("audio", 22050, voc_audio),
            accom_audio = ("audio", 22050, accom_audio),
            real_audio = ("audio", 22050, real_audio),
            fake_audio = ("audio", 22050, fake_audio),
        )
    logger.close()
    iteration += 1