# -*- coding: utf-8 -*-
#/usr/bin/python2


from __future__ import print_function

from hyperparams import Hyperparams as hp
from modules import *
import tensorflow as tf

def encoder(inputs, training=True, scope="encoder", reuse=None):
    print(inputs)
    with tf.variable_scope(scope, reuse=reuse):
        with tf.variable_scope("text_embedding"):
            tensor = embed(inputs, hp.vocab_size, hp.embed_size)  # (N, Tx, e)
        print(tensor)
        with tf.variable_scope("encoder_conv"):
            for i in range(hp.enc_layers):
                tensor = conv1d(tensor,
                                filters=hp.enc_filters,
                                kernel_size=hp.enc_kernel,
                                activation=tf.nn.relu,
                                training=training,
                                dropout_rate=hp.dropout_rate,
                                scope="encoder_conv_{}".format(i)) # (N, Tx, c)
        print(tensor)
        with tf.variable_scope("encoder_biLSTM"):
          cell = tf.nn.rnn_cell.LSTMCell(num_units=hp.enc_units)
          cell = tf.contrib.rnn.DropoutWrapper(cell, state_keep_prob=1.0-hp.z_drop)
          outputs, _  = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell,cell_bw=cell,dtype=tf.float32,inputs=tensor)
          output_fw, _ = outputs
        print(output_fw)

    return output_fw

def decoder(decoder_input, encoder_output, scope="decoder", training=True, reuse=None):

    with tf.variable_scope(scope, reuse=reuse):
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units=hp.attention_size, memory=encoder_output)
            cell = tf.contrib.rnn.GRUCell(num_units=hp.attention_size)
            attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell, attention_mechanism, attention_layer_size=hp.attention_size / 2,output_attention=True)
            # att_output = tf.contrib.rnn.OutputProjectionWrapper(
            #   attn_cell, hp.n_mels, reuse=reuse)
    print(attn_cell)

    return mel_logits, done_output

def converter(inputs, training=True, scope="converter", reuse=None):
    
    if training:
        bc_batch = hp.batch_size
    else:
        bc_batch = 1

    with tf.variable_scope(scope, reuse=reuse):

      # with tf.variable_scope("converter_rnn"):
      #   cell = tf.nn.rnn_cell.LSTMCell(num_units=hp.n_mels)
      #   outputs, _  = tf.nn.bidirectional_dynamic_rnn(
      #     cell_fw=cell,cell_bw=cell,dtype=tf.float32,inputs=inputs)
      #   output_fw, _ = outputs
      #   inputs = (inputs + output_fw) * tf.sqrt(0.5)
      #   outputs, _  = tf.nn.bidirectional_dynamic_rnn(
      #     cell_fw=cell,cell_bw=cell,dtype=tf.float32,inputs=inputs)
      #   output_fw, _ = outputs
      #   inputs = (inputs + output_fw) * tf.sqrt(0.5)
      #   output_rnn = inputs

      with tf.variable_scope("converter_conv"):
          for i in range(hp.converter_layers):
              outputs = conv_block(inputs,
                                   size=hp.converter_filter_size,
                                   rate=2**i,
                                   padding="SAME",
                                   training=training,
                                   scope="converter_conv_{}".format(i))
              inputs = (inputs + outputs) * tf.sqrt(0.5)
          output_conv = inputs
    
    # inputs = (output_rnn + output_conv) * tf.sqrt(0.5)

    with tf.variable_scope("mag_logits"):
        mag_logits = fc_block(inputs, hp.n_fft//2 + 1, training=training)

    return mag_logits