import librosa
import numpy as np
import torch

from audio_processing import STFT

##### Function to get spectrum with librosa
def read_audio_spectum(filename, n_fft, t=430):

    '''
    This is a function to get spectrogram using librosa.stft
    '''

    x, fs = librosa.load(filename)
    S = librosa.stft(x, n_fft)
    p = np.angle(S)
    
    S = np.log1p(np.abs(S[:,:t]))  
    return S, fs


##### Function to get spectrogram with Pytorch STFT
class GetSpectrogram:
    
    '''
    This is a class to define fucntion to get spectrogram with STFT
    '''

    def __init__(self, configs):
        
        # Define parameters
        self.sampling_rate = configs.sampling_rate
        self.filter_length = configs.filter_length
        self.hop_length = configs.hop_length
        self.win_length = configs.win_length
        self.max_wav_value = configs.max_wav_value
        
        # Get STFT module
        self.stft = STFT(filter_length=self.filter_length,
                         hop_length=self.hop_length,
                         win_length=self.win_length)
        
    def __call__(self, audio_path):
        
        # Get audio with normalization
        audio, _ = librosa.load(audio_path, self.sampling_rate)
        audio *= self.sampling_rate
        audio = torch.FloatTensor(audio.astype(np.float32))

        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)

        # get spectrogran
        magnitude, phase = self.stft.transform(audio_norm)

        return magnitude, phase

