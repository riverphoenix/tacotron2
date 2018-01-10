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
          output, _ = outputs
          #output = tf.concat(outputs, 2)
        print(output)

    return output

def decoder(decoder_input, encoder_output, prev_max_attentions_li=None, scope="decoder", training=True, reuse=None):

    if training:
        bc_batch = hp.batch_size
    else:
        bc_batch = 1

    with tf.variable_scope(scope, reuse=reuse):
      
      # with tf.variable_scope("attention"):
      #   attn_cell = attention(encoder_output, hp.attention_size)
      #   #attn_cell = encoder_output

      with tf.variable_scope("prenet"):
        prenet = fullyconnected(decoder_input, is_training=training, layer_size=hp.dec_prenet_size, activation=tf.nn.relu,scope='fc1')
        prenet = fullyconnected(prenet, is_training=training, layer_size=hp.dec_prenet_size, activation=tf.nn.relu,scope='fc2')

      print(prenet)

      with tf.variable_scope("decoder_conv_att"):
          # with tf.variable_scope("positional_encoding"):
          #     decoder_input_pe = embed(tf.tile(tf.expand_dims(tf.range(hp.T_y // hp.r), 0), [bc_batch, 1]),
          #              vocab_size=hp.T_y,
          #              num_units=hp.embed_size,
          #              zero_pad=False,
          #              scope="decoder_input_pe")

          #     encoder_output_pe = embed(tf.tile(tf.expand_dims(tf.range(hp.T_x), 0), [bc_batch, 1]),
          #                   vocab_size=hp.T_x,
          #                   num_units=hp.embed_size,
          #                   zero_pad=False,
          #                   scope="encoder_output_pe")

          # with tf.variable_scope("conv_att"):
          #   for i in range(hp.dec_layers):
          #       decoder_input = fc_block(decoder_input,
          #                   num_units=hp.embed_size,
          #                   dropout_rate=0 if i==0 else hp.dropout_rate,
          #                   activation_fn=tf.nn.relu,
          #                   training=training,
          #                   scope="decoder_conv_att_{}".format(i)) # (N, Ty/r, a)

          # # with tf.variable_scope("conv_att"):
          # #   _decoder_input = conv1d(decoder_input,
          # #                     filters=hp.dec_att_filters,
          # #                     kernel_size=hp.dec_att_kernel,
          # #                     activation=tf.nn.relu,
          # #                     training=training,
          # #                     dropout_rate=hp.dropout_rate,
          # #                     scope="decoder_att_conv") # (N, Tx, c)

          # #   decoder_input = (_decoder_input + decoder_input) * tf.sqrt(0.5)

          # inputs = decoder_input
          # max_attentions_li = []
          # with tf.variable_scope("att_block"):
          #   for i in range(hp.dec_layers):
          #       queries = conv_block(inputs,
          #                            size=hp.dec_filter_size,
          #                            rate=2**i,
          #                            padding="CAUSAL",
          #                            training=training,
          #                            scope="decoder_conv_block_att_{}".format(i)) # (N, Ty/r, a)

          #       inputs = (queries + inputs) * tf.sqrt(0.5)

          #       # residual connection
          #       queries = inputs + decoder_input_pe
          #       print(queries)
          #       encoder_output += encoder_output_pe
          #       print(encoder_output)

          #       # Attention Block.
          #       # tensor: (N, Ty/r, e)
          #       # alignments: (N, Ty/r, Tx)
          #       # max_attentions: (N, Ty/r)
          #       tensor, _, max_attentions = attention_block(queries,
          #                        encoder_output,
          #                        dropout_rate=hp.dropout_rate,
          #                        prev_max_attentions=prev_max_attentions_li[i],
          #                        mononotic_attention=(not training and i>2),
          #                        training=training,
          #                        scope="attention_block_{}".format(i))

          #       inputs = (tensor + queries) * tf.sqrt(0.5)
          #       max_attentions_li.append(max_attentions)
          #       print(inputs)
        attn_cell = attention_decoder(prenet, encoder_output, num_units=hp.attention_size) # (N, T', E)

          # # residual connection
          # decoder_input += decoder_input_pe
          # print(decoder_input)
          # encoder_output += encoder_output_pe
          # print(encoder_output)

          # with tf.variable_scope("att_block"):
          #   tensor, _, _ = attention_block(decoder_input,
          #                    encoder_output,
          #                    dropout_rate=hp.dropout_rate,
          #                    prev_max_attentions=prev_max_attentions_li[i],
          #                    mononotic_attention=(not training and i>2),
          #                    training=training,
          #                    scope="attention_block_{}".format(i))

          #   attn_cell = (tensor + decoder_input) * tf.sqrt(0.5)
      #attn_cell = inputs
      print(attn_cell)

      pre_att = tf.concat([prenet,attn_cell], axis=-1)
      print(pre_att)

      with tf.variable_scope("decoderLSTM"):
        cell = [tf.nn.rnn_cell.LSTMCell(size) for size in [hp.dec_LSTM_size,hp.dec_LSTM_size*2]]
        cell = tf.nn.rnn_cell.MultiRNNCell(cell)
        outputsLSTM, _ = tf.nn.dynamic_rnn(cell=cell,dtype=tf.float32, inputs=pre_att)

      print(outputsLSTM)

      LSTM_att = tf.concat([outputsLSTM,attn_cell], axis=-1)
      print(LSTM_att)

      with tf.variable_scope("projection"):
        projection = tf.layers.dense(LSTM_att,hp.n_mels)
        print(projection)

      with tf.variable_scope("postnet"):
        tensor = projection
        for i in range(hp.dec_postnet_layers):
          tensor = conv1d(tensor,
            filters=hp.dec_postnet_filters,
            kernel_size=hp.dec_postnet_size,
            activation=tf.nn.tanh if i<4 else None,
            training=training,
            dropout_rate=hp.dropout_rate,
            scope="decoder_conv_{}".format(i)) # (N, Tx, c)
        tensor = tf.layers.dense(tensor,hp.n_mels)
      print(tensor)

      mel_logits = projection + tensor
      print(mel_logits)

      if hp.include_dones:
        with tf.variable_scope("done_output"):
            done_output = fc_block(LSTM_att, 2, training=training)
            done_output = tf.nn.sigmoid(done_output)
        print(done_output)
      else:
        done_output = None

    #return mel_logits, done_output, max_attentions_li
    return mel_logits, done_output, prev_max_attentions_li

def converter(inputs, training=True, scope="converter", reuse=None):
    
    if training:
        bc_batch = hp.batch_size
    else:
        bc_batch = 1

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
    
    inputs = (output_rnn + output_conv) * tf.sqrt(0.5)

    with tf.variable_scope("mag_logits"):
        mag_logits = fc_block(inputs, hp.n_fft//2 + 1, training=training)

    return mag_logits