import torch
from librosa.filters import mel as librosa_mel_fn
from stft import STFT
import numpy as np
import librosa
class MelSpectrogram(torch.nn.Module):
	def __init__(self, hp):
		super(MelSpectrogram, self).__init__()
		self.n_mel_channels = hp.n_mel_channels
		self.sampling_rate = hp.sampling_rate
		

		self.stft_fn = STFT(hp.filter_length, hp.hop_length, hp.win_length).cuda()
		mel_basis = librosa_mel_fn(
		    hp.sampling_rate, hp.filter_length, hp.n_mel_channels,
		     hp.mel_fmin, None)

		inv_mel_basis = np.linalg.pinv(mel_basis)

		mel_basis = torch.from_numpy(mel_basis).float()
		inv_mel_basis = torch.from_numpy(inv_mel_basis).float().cuda()

		self.register_buffer('mel_basis', mel_basis)
		self.register_buffer('inv_mel_basis', inv_mel_basis)

	def _griffin_lim(self, S):
		angles =  3.1415 * (torch.rand_like(S) - 0.5)
		y = self.stft_fn.inverse(S, angles)
		y = y.squeeze(1)
		num_samples = y.size(1)
		for i in range(100):
			angles = (self.stft_fn.transform(y))[1]
			angles = angles[:,:,:S.size(2)]
			y = self.stft_fn.inverse(S, angles)
			y = y.squeeze(1)
			y = y[:,:num_samples]
		return y
	
	def transform(self, y):
		magnitudes, phases = self.stft_fn.transform(y)
		
		magnitudes = torch.abs(magnitudes)
		mel = torch.matmul(self.mel_basis, magnitudes)
		log_mel_spec = torch.log10(torch.clamp(mel, min=1e-5))

		return log_mel_spec
	def inverse(self, S):
		S = 10**(S)
		S = torch.matmul(self.inv_mel_basis, S)
		wav = self._griffin_lim(S)
		return wav