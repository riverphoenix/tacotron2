# -*- coding: utf-8 -*-
# /usr/bin/python2


from __future__ import print_function

import os, sys

from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from train import Graph
from utils import *
from data_load import load_test_data, invert_text
from scipy.io.wavfile import write
import random
import librosa
from tqdm import tqdm

def create_write_files(ret,sess,g,x,mname,cdir,typeS):

    x = np.expand_dims(x, axis=0)
    mel_output = np.zeros((1, hp.T_y // hp.r, hp.n_mels * hp.r), np.float32)
    decoder_output = np.zeros((1, hp.T_y // hp.r, hp.embed_size), np.float32)
    prev_max_attentions_li = np.zeros((hp.dec_layers, 1), np.int32)
    #for j in range(hp.T_y // hp.r):
    for j in tqdm(range(hp.T_y // hp.r)):
        _gs, _mel_output, _max_attentions_li = \
            sess.run([g.global_step, g.mel_output, g.max_attentions_li],
                     {g.x: x,
                      g.y1: mel_output,
                      g.prev_max_attentions_li:prev_max_attentions_li})
        mel_output[:, j, :] = _mel_output[:, j, :]
        #prev_max_attentions_li = np.array(_max_attentions_li)[:, :, j]
       
    #mag_output = sess.run([g.mag_output], {g.mel_output: mel_output})
    mag_output = sess.run(g.mag_output, {g.converter_input: mel_output})
    
    x = np.squeeze(x, axis=0)
    txt = invert_text(x)
    mag_output = np.squeeze(mag_output[0])

    try:
        wav = spectrogram2wav(mag_output)
        wav, _ = librosa.effects.trim(wav)
        write(cdir + "/{}mag.wav".format(mname), hp.sr, wav)
        ret.append([txt,wav,typeS+"_world"])
    except Exception:
        sys.exc_clear()

    return ret

def create_mel(sess,g,x):

    x = np.expand_dims(x, axis=0)
    mel_output = np.zeros((1, hp.T_y // hp.r, hp.n_mels * hp.r), np.float32)
    decoder_output = np.zeros((1, hp.T_y // hp.r, hp.embed_size), np.float32)
    prev_max_attentions_li = np.zeros((hp.dec_layers, 1), np.int32)
    for j in tqdm(range(hp.T_y // hp.r)):
        _gs, _mel_output, _max_attentions_li = \
            sess.run([g.global_step, g.mel_output, g.max_attentions_li],
                     {g.x: x,
                      g.y1: mel_output,
                      g.prev_max_attentions_li:prev_max_attentions_li})
        mel_output[:, j, :] = _mel_output[:, j, :]
       
    return mel_output

def create_write_files_conv(ret,sess,mel_in,g,x,mname,cdir,typeS):

    x = np.expand_dims(x, axis=0)
    mag_output = sess.run(g.mag_output, {g.converter_input: mel_in})
    
    x = np.squeeze(x, axis=0)
    txt = invert_text(x)
    mag_output = np.squeeze(mag_output[0])

    try:
        wav = spectrogram2wav(mag_output)
        wav, _ = librosa.effects.trim(wav)
        write(cdir + "/{}mag.wav".format(mname), hp.sr, wav)
        ret.append([txt,wav,typeS+"_world"])
    except Exception:
        sys.exc_clear()

    return ret

def synthesize_part(grp,config,gs,x_train,g_conv):

    x_train = random.sample(x_train, hp.batch_size)
    x_test = load_test_data()
    rand = random.randint(0,hp.batch_size-1)
    x_train = x_train[rand]
    x_test = x_test[rand]
    
    wavs = []
    if g_conv is None:
        with grp.graph.as_default():
            sv = tf.train.Supervisor(logdir=config.log_dir)
            with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                # Restore parameters
                sv.saver.restore(sess, tf.train.latest_checkpoint(config.log_dir))

                wavs = create_write_files(wavs,sess,grp,x_train,"sample_"+str(gs)+"_train_",config.log_dir,"train")
                wavs = create_write_files(wavs,sess,grp,x_test,"sample_"+str(gs)+"_test_",config.log_dir,"test")

                sess.close()
    else:
        with grp.graph.as_default():
            sv = tf.train.Supervisor(logdir=config.log_dir)
            with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                # Restore parameters
                sv.saver.restore(sess, tf.train.latest_checkpoint(config.log_dir))

                mel_out1 = create_mel(sess,grp,x_train)
                mel_out2 = create_mel(sess,grp,x_test)

                sess.close()
        with g_conv.graph.as_default():
            sv = tf.train.Supervisor(logdir=config.log_dir)
            with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                # Restore parameters
                sv.saver.restore(sess, tf.train.latest_checkpoint(config.load_converter))

                wavs = create_write_files_conv(wavs,sess,mel_out1,g_conv,x_train,"sample_"+str(gs)+"_train_",config.log_dir,"train")
                wavs = create_write_files_conv(wavs,sess,mel_out2,g_conv,x_test,"sample_"+str(gs)+"_test_",config.log_dir,"test")

                sess.close()

    return wavs