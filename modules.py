# -*- coding: utf-8 -*-
#/usr/bin/python2


from __future__ import print_function, division

from hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np
from zoneout_LSTM import ZoneoutLSTMCell
from tensorflow.contrib.rnn import LSTMStateTuple

def embed(inputs, vocab_size, num_units, zero_pad=False, scope="embedding", reuse=None):
    
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table', 
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.5))
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]), 
                                      lookup_table[1:, :]), 0)

        outputs = tf.nn.embedding_lookup(lookup_table, inputs)

    return outputs
 
def glu(inputs):
    a, b = tf.split(inputs, 2, -1)
    outputs = a * tf.nn.sigmoid(b)
    return outputs

def conv1d(inputs, kernel_size, filters, activation, dropout_rate, training=False, scope="conv1d", reuse=None):
  
  with tf.variable_scope(scope, reuse=reuse):
    conv1d_output = tf.layers.conv1d(
      inputs,
      filters=filters,
      kernel_size=(kernel_size,),
      padding='same')

    batched = tf.layers.batch_normalization(conv1d_output, training=training)

    if activation is not None:
      activated = activation(batched)
    else:
      activated = batched

    return tf.layers.dropout(activated, rate=dropout_rate, training=training)

def conv_block(inputs,
               num_units=None,
               size=5,
               rate=1,
               padding="SAME",
               dropout_rate=0,
               training=False,
               scope="conv_block",
               reuse=None):

    in_dim = inputs.get_shape().as_list()[-1]
    if num_units is None: num_units = in_dim

    with tf.variable_scope(scope, reuse=reuse):
        inputs = tf.layers.dropout(inputs, rate=dropout_rate, training=training)

        if padding.lower() == "causal":
            pad_len = (size - 1) * rate  # padding size
            inputs = tf.pad(inputs, [[0, 0], [pad_len, 0], [0, 0]])
            padding = "VALID"

        V = tf.get_variable('V',
                            shape=[size, in_dim, num_units*2],
                            dtype=tf.float32) # (width, in_dim, out_dim)
        g = tf.get_variable('g',
                            shape=(num_units*2,),
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.variance_scaling_initializer(factor=(4.*(1.-dropout_rate))/size))
        b = tf.get_variable('b',
                            shape=(num_units*2,),
                            dtype=tf.float32,
                            initializer=tf.zeros_initializer)

        if hp.normalization:
          V_norm = tf.nn.l2_normalize(V, [0, 1])  # (width, in_dim, out_dim)
        else:
          V_norm = V
        W = V_norm * tf.reshape(g, [1, 1, num_units*2])

        outputs = tf.nn.convolution(inputs, W, padding, dilation_rate=[rate]) + b
        outputs = glu(outputs)

    return outputs

def fc_block(inputs,
             num_units,
             dropout_rate=0,
             activation_fn=None,
             training=False,
             scope="fc_block",
             reuse=None):

    _, T, in_dim = inputs.get_shape().as_list()
    with tf.variable_scope(scope, reuse=reuse):
        inputs = tf.layers.dropout(inputs, rate=dropout_rate, training=training)

        # Transformation
        V = tf.get_variable('V',
                            shape=[in_dim, num_units],
                            dtype=tf.float32)
        g = tf.get_variable('g',
                            shape=(num_units),
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.variance_scaling_initializer(
                                factor=(1. - dropout_rate)))
        b = tf.get_variable('b', shape=(num_units), dtype=tf.float32, initializer=tf.zeros_initializer)

        if hp.normalization:
          V_norm = tf.nn.l2_normalize(V, [0])
        else:
          V_norm = V
        W = V_norm * tf.expand_dims(g, 0)

        outputs = tf.matmul(tf.reshape(inputs, (-1, in_dim)), W) + b
        outputs = tf.reshape(outputs, (-1, T, num_units))

        if activation_fn is not None:
            outputs = activation_fn(outputs)

    return outputs

def fullyconnected(inputs, is_training, layer_size, activation,scope='fc',reuse=None):
  with tf.variable_scope(scope):
    dense = tf.layers.dense(inputs, units=layer_size, activation=activation)
    output = tf.layers.dropout(dense, rate=hp.dropout_rate, training=is_training)
  return output


def bidirectional_LSTM(inputs, scope, training):

  with tf.variable_scope(scope):
    outputs, (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(
      # tf.nn.rnn_cell.LSTMCell(hp.enc_units),
      # tf.nn.rnn_cell.LSTMCell(hp.enc_units),
      ZoneoutLSTMCell(hp.enc_units, training, zoneout_factor_cell=hp.z_drop, zoneout_factor_output=hp.z_drop,),
      ZoneoutLSTMCell(hp.enc_units, training, zoneout_factor_cell=hp.z_drop, zoneout_factor_output=hp.z_drop,),
      inputs, dtype=tf.float32)

    #Concatenate c states and h states from forward
    #and backward cells
    encoder_final_state_c = tf.concat( (fw_state.c, bw_state.c), 1)
    encoder_final_state_h = tf.concat( (fw_state.h, bw_state.h), 1)

    #Get the final state to pass as initial state to decoder
    final_state = LSTMStateTuple( c=encoder_final_state_c, h=encoder_final_state_h)

  return tf.concat(outputs, axis=2), final_state # Concat forward + backward outputs and final states


def unidirectional_LSTM(is_training, layers, size):
  
  # rnn_layers = [tf.nn.rnn_cell.LSTMCell(hp.enc_units) for i in range(layers)]

  rnn_layers = [ZoneoutLSTMCell(size, is_training, zoneout_factor_cell=hp.z_drop,
                           zoneout_factor_output=hp.z_drop,
                           ext_proj=hp.n_mels) for i in range(layers)]

  stacked_LSTM_Cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
  return stacked_LSTM_Cell        