import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils import load_wav_to_torch
import os

def make_inf_iterator(data_iterator):
    while True:
        for data in data_iterator:
            yield data

class AudioLoader(torch.utils.data.Dataset):
    """
        1) loads audio
    """
    def __init__(self, audio_path):
        self.audio_path = audio_path
        self.audios = []
        for root, dirs, files in os.walk(audio_path):
            for f in files:
                self.audios += [os.path.join(root, f)]
        random.seed(1234)
        random.shuffle(self.audios)

    def __getitem__(self, index):
        item = self.audios[index]
        return load_wav_to_torch(item)[0]

    def __len__(self):
        return len(self.audios)


class AudioCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """

    def __call__(self, batch):
        """Collate's training batch from audio
        PARAMS
        ------
        batch: [audio]
        """
        '''
        for i in range(len(batch)):
            if batch[i].shape[1] != 861:
                batch[i] = batch[i - 1]
        '''
        return torch.tensor(batch)#torch.stack(batch, dim = 0)


class AudioNpyLoader(torch.utils.data.Dataset):
    """
        1) loads audio
    """
    def __init__(self, audio_path):
        self.audio_path = audio_path
        self.audios = os.listdir(self.audio_path)
        
        random.seed(1234)
        random.shuffle(self.audios)

    def __getitem__(self, index):
        item = f'{self.audio_path}/{self.audios[index]}'
        return np.load(item)

    def __len__(self):
        return len(self.audios)