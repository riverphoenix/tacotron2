# -*- coding: utf-8 -*-
#/usr/bin/python2

from __future__ import print_function

import numpy as np
import librosa
import copy
import matplotlib
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.pyplot import step, show
import librosa.display

from hyperparams import Hyperparams as hp


def spectrogram2wav(mag):
    '''# Generate wave file from spectrogram'''
    # transpose
    mag = mag.T

    # de-noramlize
    mag = (np.clip(mag, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db

    # to amplitude
    mag = librosa.db_to_power(mag)

    # wav reconstruction
    wav = griffin_lim(mag)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -hp.preemphasis], wav)

    # trim
    wav, _ = librosa.effects.trim(wav)

    return wav

def griffin_lim(spectrogram):
    X_best = copy.deepcopy(spectrogram)
    for i in range(hp.n_iter):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, hp.n_fft, hp.hop_length, win_length=hp.win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)

    return y

def invert_spectrogram(spectrogram):

    return librosa.istft(spectrogram, hp.hop_length, win_length=hp.win_length, window="hann")


def plot_losses(config,Kmel_out,Ky1,Kdone_out,Ky2,Kmag_out,Ky3,gs):
   
    plt.figure(figsize=(10, 10))

    ax1 = plt.subplot(3, 2, 1)
    librosa.display.specshow(Kmel_out[0,:,:].T,y_axis='linear')
    plt.title('Predicted mel')
    plt.colorbar()
    plt.tight_layout()

    ax2 = plt.subplot(3, 2, 2,sharey=ax1)
    librosa.display.specshow(Ky1[0,:,:].T,y_axis='linear')
    plt.title('Original mel')
    plt.colorbar()
    plt.tight_layout()

    ax3 = plt.subplot(3, 2, 3)
    librosa.display.specshow(Kmag_out[0,:,:].T,y_axis='linear')
    plt.title('Predicted mag')
    plt.colorbar()
    plt.tight_layout()

    ax4 = plt.subplot(3, 2, 4,sharey=ax3)
    librosa.display.specshow(Ky3[0,:,:].T,y_axis='linear')
    plt.title('Original mag')
    plt.colorbar()
    plt.tight_layout()

    KDone = Kdone_out[0,:,:]
    Kd = []
    for i in range(KDone.shape[0]):
        if KDone[i,0] > KDone[i,1]:
            Kd.append(0)
        else:
            Kd.append(1)

    ind = np.arange(len(Kd))
    width = 1.0

    ax5 = plt.subplot(3, 2, 5)
    ax5.bar(ind, Kd, width, color='r')
    plt.title('Predicted Dones')
    plt.tight_layout()
  
    ax6 = plt.subplot(3, 2, 6)
    ax6.bar(ind, Ky2[0,:], width, color='r')
    plt.title('Original Dones')
    plt.tight_layout()

    plt.savefig('{}/losses_{}.png'.format(config.log_dir, gs), format='png')

    plt.close('all')

def plot_wavs(config,wavs,gs):
    if len(wavs)!=0:
        plt.figure(figsize=(10, 10))
        for i in range(len(wavs)):
            wav = wavs[i]
            txt = str(wav[2])+':'+str(wav[0])
            wv = wav[1]

            plt.subplot(len(wavs),1, i+1)
            librosa.display.waveplot(wv, sr=hp.sr)
            plt.title(txt)
        plt.savefig('{}/wavs_{}.png'.format(config.log_dir, gs), format='png')

        plt.close('all')