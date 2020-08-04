import sys
import os
import argparse
import warnings
import yaml
import pickle
import librosa
import librosa.core as core
from librosa.filters import mel as librosa_mel_fn
import scipy
warnings.filterwarnings('ignore')

import numpy as np
import random
from random import randint

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F

sys.path.append('model/')
from layers import *
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

sys.path.append('utils/')
#from dataloader import VocDataset, get_voc_datasets
from hparams import *
from save_and_load import *
from process_yaml_model import YamlModelProcesser
from optim_step import *
from utils import smooth
from scipy.signal import savgol_filter

sys.path.append('logger')
from logger import Logger
from logger_utils import prepare_directories_and_logger
from plotting_utils import  plot_spectrogram_to_numpy

sys.path.append('loss')
from gradient_penalty import gradient_penalty
from began_loss import BEGANRecorder, BEGANLoss

#sys.path.append('VocalMelodyExtPatchCNN')
#from MelodyExt import melody_extraction
sys.path.append('vocoder/melgan-neurips')
from mel2wav.interface import MelVocoder
from scipy.interpolate import interp1d

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
parser.add_argument('--feat_type', type=str, default="mag",
                    required = False, help='data_type')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.rank
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
## hyperparamerter
hp = create_hparams(f"hp_config/{args.hp_config}")

## create logger
logger = prepare_directories_and_logger(Logger, output_directory = f'output/{args.output_directory}')

mel_basis = librosa_mel_fn(
            22050, 1024, 80, 0, None)

inv_mel_basis = np.linalg.pinv(mel_basis)

####################################################################
"""
   Data Loader Part
"""
def make_inf_iterator(data_iterator):
    while True:
        for data in data_iterator:
            yield data
class AudioLoader(torch.utils.data.Dataset):
    """
        1) loads audio
        2) computes mel-spectrograms from audio files.
        (optional) 3) computes f0 and beat informations
    """
    def __init__(self, id_list):
        self.id_list = id_list
        
        random.shuffle(self.id_list)
        self.sr = 22050

        self.nfft = 1024
        self.wlen = 1024
        self.hop = 256

    def __getitem__(self, index):
        audio_list = []
        tf = self.id_list[index]
        read_file = self.get_read(tf)

        read_audio = f'{read_file[:-4]}.wav'
        song_audio = f'{tf[:-4]}.wav'
        
        with open(tf, 'rb') as f:
            song_txt = f.read().splitlines()
        with open(read_file, 'rb') as f:
            read_txt = f.read().splitlines()

        melody_name = "_".join([tf.split('/')[-3],
                            tf.split('/')[-1]])[:-4]
        melody = np.load(f'../sp2si-code/melody_contour/{melody_name}.npy')
        
        index_begin = randint(0,len(read_txt) - 40) + 1
        index_end = index_begin + 30


        song_begin = float(song_txt[index_begin].split()[0])
        song_end = float(song_txt[index_end].split()[1])
        

        song_dur = song_end - song_begin
        while song_dur < 7 and index_end < len(read_txt) - 2:
            index_end += 1
            song_end = float(song_txt[index_end].split()[1])
            song_dur = song_end - song_begin
        while song_dur > 10 :
            index_end -= 1
            song_end = float(song_txt[index_end].split()[1])
            song_dur = song_end - song_begin
        
        read_begin = float(read_txt[index_begin].split()[0])
        read_end = float(read_txt[index_end].split()[1])
        read_dur = read_end - read_begin

        
        read_audio = core.load(read_audio, sr=self.sr, mono=True, offset=read_begin, duration=read_dur)[0]
        song_audio = core.load(song_audio, sr=self.sr, mono=True, offset=song_begin, duration=song_dur)[0]

        index_sing_dur = []
        index_read_dur = []
        t = self.sr
        rsi = float(read_txt[index_begin].split()[0])
        ssi = float(song_txt[index_begin].split()[0])
        i = index_begin
        while i < index_end+1:
            s_begin, s_end, _ = song_txt[i].split()
            r_begin, r_end, _ = read_txt[i].split()
            s_begin, s_end, r_begin, r_end = float(s_begin), float(s_end), float(r_begin), float(r_end)
            while (s_end - s_begin< 0.2 or r_end - r_begin < 0.2) and i<index_end + 1:
                i = i + 1
                _, s_end, _ = song_txt[i].split()
                _, r_end, _ = read_txt[i].split()
                s_end , r_end = float(s_end), float(r_end)
            index_read_dur += [((r_begin-rsi)*t , (r_end-rsi)*t) ]
            index_sing_dur += [((s_begin-ssi)*t  , (s_end-ssi)*t)]
            i = i + 1
        
        read_audio_list = []
        for i in range(len(index_read_dur)):
            r_begin, r_end = index_read_dur[i]
            s_begin, s_end = index_sing_dur[i]
            if r_end - r_begin ==0 or s_end - s_begin == 0:
                read_audio_list += [read_audio[int(r_begin):int(r_end)]]
                continue
            rate = (s_end - s_begin)/(r_end - r_begin)
            read_audio_slice = librosa.effects.time_stretch(read_audio[int(r_begin):int(r_end)], 1/rate)
            read_audio_list += [read_audio_slice]
        
        read_audio = np.concatenate(read_audio_list, axis=0)

        read_stft = core.stft(read_audio, n_fft=self.nfft, hop_length=self.hop, win_length=self.wlen)
        
        song_stft = core.stft(song_audio, n_fft=self.nfft, hop_length=self.hop, win_length=self.wlen)
        
        song_stft = song_stft[...,:-1]
        
        rate = read_stft.shape[1] / song_stft.shape[1]
        read_stft = core.phase_vocoder(read_stft, rate, self.hop)
        read_stft = read_stft[:, :song_stft.shape[1]]
        
        read_stft = abs(read_stft)
        song_stft = abs(song_stft)
        
        if args.feat_type == "mel":
            read_stft = np.matmul(mel_basis, read_stft)
            song_stft = np.matmul(mel_basis, song_stft)
            
        read_mag = np.log10(np.clip((read_stft), a_min=1e-5, a_max=100000))
        song_mag = np.log10(np.clip((song_stft), a_min=1e-5, a_max=100000))
        
        return song_mag, read_mag,  read_audio

    def get_read(self,song_file):
        speech_file = song_file.split('/')
        speech_file[-2] = 'read'
        return '/'.join(speech_file)

    def __len__(self):
        return len(self.id_list)


class AudioCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """

    def __call__(self, batch):
        """Collate's training batch from audio
        PARAMS
        ------
        batch: [audio]
        """

        # sort audio with their length
        #print (batch[0].shape)
        l = list(zip(*batch))
        song, read,  read_real = l[0], l[1], l[2]
        
        index = sorted(range(len(song)), key=lambda k: -song[k].shape[1])
        
        
        song = [song[i] for i in index ]
        read = [read[i] for i in index]
        read_real = [read_real[i] for i in index]
        maxn = song[0].shape[1]
        for i in range(len(song)):
            
            song[i] = np.pad(song[i],((0,0),(0,maxn - song[i].shape[1])), 'reflect')
            
            read[i] = np.pad(read[i],((0,0),(0,maxn - read[i].shape[1])), 'reflect')
            
            song[i] = torch.from_numpy(song[i]).unsqueeze(0)
            read[i] = torch.from_numpy(read[i]).unsqueeze(0)
            
        
        return torch.cat(song, dim = 0), torch.cat(read, dim = 0), read_real



################################################################################
from os import listdir


feat_dim = 512
with open("/home/ericwudayi/nas189/homes/ericwudayi/NUS/dataset.pkl",'rb') as f:
    dataset = pickle.load(f)
with open("/home/ericwudayi/nas189/homes/ericwudayi/NUS/dataset_test.pkl",'rb') as f:
    dataset_test = pickle.load(f)
'''
for i in range(len(dataset)):
    dataset[i] = dataset[i].split("/")
    dataset[i] = dataset[i][:3] + ['remote192'] + dataset[i][3:]
    dataset[i] = '/'.join(dataset[i])
    
for i in range(len(dataset_test)):
    dataset_test[i] = dataset_test[i].split("/")
    dataset_test[i] = dataset_test[i][:3] + ['remote192'] + dataset_test[i][3:]
    dataset_test[i] = '/'.join(dataset_test[i])
'''
dataset = AudioLoader(dataset)
dataset_test = AudioLoader(dataset_test)

iterator_tr = DataLoader(
        dataset,
        batch_size=4,
        num_workers=2,
        shuffle=True,
        drop_last=True,
        pin_memory=True, collate_fn = AudioCollate())

inf_iterator_tr_speech = make_inf_iterator(iterator_tr)

iterator_test = DataLoader(
        dataset_test,
        batch_size=1,
        num_workers=1,
        shuffle=True,
        drop_last=True,
        pin_memory=True, collate_fn = AudioCollate())
inf_iterator_test_speech = make_inf_iterator(iterator_test)

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
    Optimizer
"""
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
##########################################################

'''
    ### Vocoder block ###
    MelGan vocoder, vocoder/modules
'''

vocoder = MelVocoder(path = "vocoder/melgan-neurips/scripts/logs/NUS")

#######################################################################

while True:
    
    
    song_padded, read_padded,  read_real = \
        next(inf_iterator_tr_speech)
    song_padded, read_padded = \
    song_padded.float().cuda(), read_padded.float().cuda()
    

    song_padded = song_padded[...,:song_padded.size(2)//64 * 64]
    read_padded = read_padded[...,:read_padded.size(2)//64 * 64]
    
    print ("song_padded: ", song_padded.size())
    print ("read_padded: ", read_padded.size())

    fake_song_padded = read_padded
    
    for (i,block) in enumerate(m):
        if i == 0:
            fake_song_padded = block(fake_song_padded)
        else:
            fake_read_padded = block(fake_song_padded)

    #song_padded_auto = auto(song_padded)
    loss_sp2sing = 0.1 * criterion(fake_song_padded, song_padded)
    loss_sp2sing = loss_sp2sing.mean()
    
    loss_cycle = 0.1 * criterion(fake_read_padded, read_padded).mean()
      
    
    loss_gan, loss_dis, real_dloss, fake_dloss = BEGANLoss(dis_high, song_padded, fake_song_padded, k)#song_len_padded[...,:fake_song_padded.size(2)]
    OptimStep([(m, opt, loss_gan +loss_sp2sing + loss_cycle, True),
        (dis_high, opt_dis, loss_dis, False)], 3) #(auto, opt_auto, loss_auto + 0.01*latent_loss, False),
    k, convergence = recorder(real_dloss, fake_dloss, update_k=True)
    
    if iteration % 5 == 0:
        logger.log_training(iteration = iteration, loss_gan = loss_gan,loss_cycle = loss_cycle,
        loss_dis = loss_dis, loss_sing2sp = loss_sp2sing,
        k = k, convergence = convergence)
    
    if (iteration % 50 == 0):

        save_checkpoint(m, opt, iteration, f'checkpoint/{args.checkpoint_path}/gen')
        save_checkpoint(dis_high, opt_dis, iteration, f'checkpoint/{args.checkpoint_path}/dis')
        
        idx = 0
        fake_song_padded = torch.clamp(fake_song_padded,-10000,1.8)
        singing_melgan = fake_song_padded[idx:idx+1]
        singing_real_melgan = song_padded[idx:idx+1]
        fake_singing_audio = vocoder.inverse(singing_melgan).detach().cpu().numpy()[0]
        real_singing_audio = vocoder.inverse(singing_real_melgan).detach().cpu().numpy()[0]
        logger.log_validation(iteration = iteration,
            fake_train_singing_audio = ("audio", 22050, fake_singing_audio),
            real_train_singing_audio = ("audio", 22050, real_singing_audio),
            real_train_speech_audio = ("audio", 22050, read_real[idx]),
        )

        

        with torch.set_grad_enabled(False):
            
            m.eval()

            song_padded, read_padded, read_real = \
                next(inf_iterator_test_speech)
            song_padded, read_padded = \
            song_padded.float().cuda(), read_padded.float().cuda(),

            song_padded = song_padded[...,:song_padded.size(2)//64 * 64]
            read_padded = read_padded[...,:read_padded.size(2)//64 * 64]
            fake_song_padded = read_padded

            for (i,block) in enumerate(m):
                if i == 0:
                    fake_song_padded = block(fake_song_padded)
                else:
                    fake_read_padded = block(fake_song_padded)

            
            idx = 0
            fake_song_padded = torch.clamp(fake_song_padded,-10000,1.8)
            
            singing_melgan = fake_song_padded[idx:idx+1]
            singing_real_melgan = song_padded[idx:idx+1]
            
            fake_singing_audio = vocoder.inverse(singing_melgan).detach().cpu().numpy()[0]
            real_singing_audio = vocoder.inverse(singing_real_melgan).detach().cpu().numpy()[0]
            
            logger.log_validation(iteration = iteration,
                mel_singing = ("image", plot_spectrogram_to_numpy(), song_padded[idx]),
                mel_singing_generate = ("image", plot_spectrogram_to_numpy(), fake_song_padded[idx]),
                mel_speech = ("image", plot_spectrogram_to_numpy(), read_padded[idx]),
                
                fake_singing_audio = ("audio", 22050, fake_singing_audio),
                real_singing_audio = ("audio", 22050, real_singing_audio),
                real_speech_audio = ("audio", 22050, read_real[idx]),
            )
            m.train()
            logger.close()
    iteration += 1
    
