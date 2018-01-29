# -*- coding: utf-8 -*-
#/usr/bin/python2


from __future__ import print_function

from hyperparams import Hyperparams as hp
from modules import *
import tensorflow as tf
from rnn_wrappers import TacotronDecoderWrapper
from attention_wrapper import AttentionWrapper, LocationBasedAttention, BahdanauAttention
from helpers import TacoTrainingHelper, TacoTestHelper
from dynamic_decoder import dynamic_decode
from custom_decoder import CustomDecoder


def encoder(inputs, training=True, scope="encoder", reuse=None):
    if hp.print_shapes: print(inputs)
    with tf.variable_scope(scope, reuse=reuse):
        with tf.variable_scope("text_embedding"):
            tensor = embed(inputs, hp.vocab_size, hp.embed_size)  # (N, Tx, e)
        if hp.print_shapes: print(tensor)
        with tf.variable_scope("encoder_conv"):
            for i in range(hp.enc_layers):
                tensor = conv1d(tensor,
                                filters=hp.enc_filters,
                                kernel_size=hp.enc_kernel,
                                activation=tf.nn.relu,
                                training=training,
                                dropout_rate=hp.dropout_rate,
                                scope="encoder_conv_{}".format(i)) # (N, Tx, c)
        if hp.print_shapes: print(tensor)

        output, _ = bidirectional_LSTM(tensor, 'encoder_biLSTM', training=training)
        if hp.print_shapes: print(output)

    return output

def decoder(mel_targets, encoder_output, scope="decoder", training=True, reuse=None):

    with tf.variable_scope(scope, reuse=reuse):
      
      decoder_cell = TacotronDecoderWrapper(unidirectional_LSTM(training, layers=hp.dec_LSTM_layers, size=hp.dec_LSTM_size), training)

      attention_decoder = AttentionWrapper(
        decoder_cell,
        LocationBasedAttention(hp.attention_size, encoder_output),
        #BahdanauAttention(hp.attention_size, encoder_output),
        alignment_history=True,
        output_attention=False)

      decoder_state = attention_decoder.zero_state(batch_size=hp.batch_size, dtype=tf.float32)
      projection = tf.tile([[0.0]], [hp.batch_size, hp.n_mels])
      final_projection =tf.zeros([hp.batch_size, hp.T_y//hp.r, hp.n_mels], tf.float32)
      if hp.include_dones:
        LSTM_att =tf.zeros([hp.batch_size, hp.T_y//hp.r, hp.dec_LSTM_size*2], tf.float32)
      else:
        LSTM_att = 0
      step = 0

      def att_condition(step, projection, final_projection, decoder_state, mel_targets,LSTM_att):
        return step <  hp.T_y//hp.r

      def att_body(step, projection, final_projection, decoder_state, mel_targets,LSTM_att):
        if training:
          if step == 0:
            projection, decoder_state, _, LSTM_next = attention_decoder.call(tf.tile([[0.0]], [hp.batch_size, hp.n_mels]), decoder_state)
          else:
            projection, decoder_state, _, LSTM_next = attention_decoder.call(mel_targets[:, step-1, :], decoder_state)
        else:
          projection, decoder_state, _, LSTM_next = attention_decoder.call(projection, decoder_state)
        fprojection = tf.expand_dims(projection,axis=1)
        final_projection = tf.concat([final_projection,fprojection],axis=1)[:,1:,:]
        if hp.include_dones:
          fLSTM_next = tf.expand_dims(LSTM_next,axis=1)
          LSTM_att = tf.concat([LSTM_att,fLSTM_next],axis=1)[:,1:,:]
        return ((step+1), projection, final_projection, decoder_state, mel_targets,LSTM_att)
        
      res_loop = tf.while_loop(att_condition, att_body,
        loop_vars=[step, projection, final_projection, decoder_state, mel_targets,LSTM_att],
        parallel_iterations=hp.parallel_iterations, swap_memory=False)

      final_projection = res_loop[2]
      final_decoder_state = res_loop[3]
      concat_LSTM_att = res_loop[5]
      step = res_loop[0]

      if hp.print_shapes: print(final_projection)

      with tf.variable_scope("postnet"):
        tensor = final_projection
        for i in range(hp.dec_postnet_layers):
          tensor = conv1d(tensor,
            filters=hp.dec_postnet_filters,
            kernel_size=hp.dec_postnet_size,
            activation=tf.nn.tanh if i<4 else None,
            training=training,
            dropout_rate=hp.dropout_rate,
            scope="decoder_conv_{}".format(i)) # (N, Tx, c)
        #tensor = tf.layers.dense(tensor,hp.n_mels)
        tensor = tf.contrib.layers.fully_connected(tensor, hp.n_mels, activation_fn=None, biases_initializer=tf.zeros_initializer())
      if hp.print_shapes: print(tensor)

      mel_logits = final_projection + tensor
      if hp.print_shapes: print(mel_logits)

      if hp.include_dones:
        with tf.variable_scope("done_output"):
            done_output = fc_block(concat_LSTM_att, 2, training=training)
            done_output = tf.nn.sigmoid(done_output)
        if hp.print_shapes: print(done_output)
      else:
        done_output = None
        concat_LSTM_att = None

    return mel_logits, final_projection, done_output, final_decoder_state, concat_LSTM_att,step

def converter(inputs, training=True, scope="converter", reuse=None):

    with tf.variable_scope(scope, reuse=reuse):

      with tf.variable_scope("converter_rnn"):
        cell = tf.nn.rnn_cell.LSTMCell(num_units=hp.n_mels)
        outputs, _  = tf.nn.bidirectional_dynamic_rnn(
          cell_fw=cell,cell_bw=cell,dtype=tf.float32,inputs=inputs)
        output_fw, _ = outputs
        inputs = (inputs + output_fw) * tf.sqrt(0.5)
        outputs, _  = tf.nn.bidirectional_dynamic_rnn(
          cell_fw=cell,cell_bw=cell,dtype=tf.float32,inputs=inputs)
        output_fw, _ = outputs
        inputs = (inputs + output_fw) * tf.sqrt(0.5)
        output_rnn = inputs
        if hp.print_shapes: print(output_rnn)

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
          if hp.print_shapes: print(output_conv)
    
    inputs = (output_rnn + output_conv) * tf.sqrt(0.5)
    if hp.print_shapes: print(inputs)

    with tf.variable_scope("mag_logits"):
        mag_logits = fc_block(inputs, hp.n_fft//2 + 1, training=training)
        if hp.print_shapes: print(mag_logits)

    return mag_logits