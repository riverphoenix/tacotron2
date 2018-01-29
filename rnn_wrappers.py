# -*- coding: utf-8 -*-
#/usr/bin/python2

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from hyperparams import Hyperparams as hp
from modules import *

class TacotronDecoderWrapper(RNNCell):
  
  def __init__(self, cell, is_training):
    super(TacotronDecoderWrapper, self).__init__()
    self._cell = cell
    self._is_training = is_training

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    #return (self.batch_size, hparams.num_mels)
    return self._cell.output_size

  def call(self, inputs, state):
    #Get context vector from cell state
    context_vector = state.attention
    cell_state = state.cell_state

    #Compute prenet output
    prenet = fullyconnected(inputs, is_training=self._is_training, layer_size=hp.dec_prenet_size, activation=tf.nn.relu,scope='fc1')
    prenet_outputs = fullyconnected(prenet, is_training=self._is_training, layer_size=hp.dec_prenet_size, activation=tf.nn.relu,scope='fc2')

    #Concat prenet output and context vector
    concat_output_prenet = tf.concat([prenet_outputs, context_vector], axis=-1)

    #Compute LSTM output
    LSTM_output, next_cell_state = self._cell(concat_output_prenet, cell_state)

    #Concat LSTM output and context vector
    concat_output_LSTM = tf.concat([LSTM_output, context_vector], axis=-1)

    #Linear projection
    cell_output = tf.contrib.layers.fully_connected(concat_output_LSTM, hp.n_mels, activation_fn=None, biases_initializer=tf.zeros_initializer(), scope='decoder_projection_layers')
    cell_output = (cell_output,LSTM_output)

    return cell_output, next_cell_state, concat_output_LSTM

  def zero_state(self, batch_size, dtype):
    return self._cell.zero_state(batch_size, dtype)