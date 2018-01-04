# -*- coding: utf-8 -*-
# #/usr/bin/python2

import numpy as np
import librosa

from hyperparams import Hyperparams as hp
import glob
import os
import tqdm

import multiprocessing

def get_spectrograms(sound_file):
    # Loading sound file
    y, sr = librosa.load(sound_file, sr=hp.sr)

    # Trimming
    y, _ = librosa.effects.trim(y)

    # Preemphasis
    y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])

    # Scale between -1 and 1
    y = y / np.abs(y).max()

    # stft
    stft = librosa.stft(y=y, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length)

    # magnitude spectrogram
    mag, _ = librosa.magphase(stft)

    # mel spectrogram
    mel = librosa.feature.melspectrogram(S=mag,n_mels=hp.n_mels,fmin=hp.lowcut, fmax=hp.highcut)

    # Sequence length
    done = np.ones_like(mel[0, :]).astype(np.int32)

    # to decibel
    mel = librosa.amplitude_to_db(mel)
    mag = librosa.power_to_db(mag)

    # Normalize
    mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 0, 1)
    mag = np.clip((mag - hp.ref_db + hp.max_db) / hp.max_db, 0, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

    return mel, done, mag

def prep_all_files(files):

    for file in tqdm.tqdm(files):
        fname = os.path.basename(file)
        
        mel, done, mag = get_spectrograms(file)
        #mel, mag = get_spectrograms(file)
        np.save(os.path.join(mel_folder, fname.replace(".wav", ".npy")), mel)
        np.save(os.path.join(mag_folder, fname.replace(".wav", ".npy")), mag)
        np.save(os.path.join(done_folder, fname.replace(".wav", ".npy")), done)

def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
             for i in range(wanted_parts) ]

if __name__ == "__main__":
    wav_folder = os.path.join(hp.data, 'wavs')
    mel_folder = os.path.join(hp.data, 'mels')
    mag_folder = os.path.join(hp.data, 'mags')
    done_folder = os.path.join(hp.data, 'dones')

    for folder in (mel_folder, mag_folder, done_folder):
    #for folder in (mel_folder, mag_folder):
        if not os.path.exists(folder): os.mkdir(folder)

    files = glob.glob(os.path.join(wav_folder, "*"))
    if hp.prepro_gpu > 1:
        files = split_list(files, wanted_parts=hp.prepro_gpu)
        for i in range(hp.prepro_gpu):
            p = multiprocessing.Process(target=prep_all_files, args=(files[i],))
            p.start()
    else:
        prep_all_files(files)