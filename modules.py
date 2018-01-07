# -*- coding: utf-8 -*-
#/usr/bin/python2


from __future__ import print_function, division

from hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np

def embed(inputs, vocab_size, num_units, zero_pad=True, scope="embedding", reuse=None):
    
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table', 
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
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
    inputs = tf.layers.dropout(inputs, rate=dropout_rate, training=training)

    conv1d_output = tf.layers.conv1d(
      inputs,
      filters=filters,
      kernel_size=kernel_size,
      activation=activation,
      padding='same')

    return tf.layers.batch_normalization(conv1d_output, training=training)

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
  drop_rate = hp.dropout_rate if is_training else 0.0
  with tf.variable_scope(scope):
    dense = tf.layers.dense(inputs, units=layer_size, activation=activation)
    output = tf.layers.dropout(dense, rate=drop_rate)
  return output

def positional_encoding(inputs,
                        num_units,
                        position_rate=1.,
                        zero_pad=True,
                        scale=True,
                        scope="positional_encoding",
                        reuse=None):

    N, T = inputs.get_shape().as_list()
    with tf.variable_scope(scope, reuse=reuse):
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos*position_rate / np.power(10000, 2.*i/num_units) for i in range(num_units)]
            for pos in range(T)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        # Convert to a tensor
        lookup_table = tf.convert_to_tensor(position_enc, tf.float32)

        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

        if scale:
            outputs *= num_units**0.5

        return outputs

def attention_block(queries,
                    keys,
                    vals,
                    dropout_rate=0,
                    prev_max_attentions=None,
                    training=False,
                    mononotic_attention=False,
                    scope="attention_block",
                    reuse=None):

    _keys = keys
    with tf.variable_scope(scope, reuse=reuse):

        with tf.variable_scope("query_proj"):
            queries = fc_block(queries, hp.attention_size, training=training)

        with tf.variable_scope("key_proj"):
            keys = fc_block(keys, hp.attention_size, training=training)

        with tf.variable_scope("value_proj"):
            vals = fc_block(vals, hp.attention_size, training=training)

        with tf.variable_scope("alignments"):
            attention_weights = tf.matmul(queries, keys, transpose_b=True)

            _, Ty, Tx = attention_weights.get_shape().as_list()
            if mononotic_attention: # for inference
                key_masks = tf.sequence_mask(prev_max_attentions, Tx)
                reverse_masks = tf.sequence_mask(Tx - hp.attention_win_size - prev_max_attentions, Tx)[:, ::-1]
                masks = tf.logical_or(key_masks, reverse_masks)
                masks = tf.tile(tf.expand_dims(masks, 1), [1, Ty, 1])
                paddings = tf.ones_like(attention_weights) * (-2 ** 32 + 1)
                attention_weights = tf.where(tf.equal(masks, False), attention_weights, paddings)
            alignments = tf.nn.softmax(attention_weights)
            max_attentions = tf.argmax(alignments, -1)

        with tf.variable_scope("context"):
            ctx = tf.layers.dropout(alignments, rate=dropout_rate, training=training)
            ctx = tf.matmul(ctx, vals)
            ctx *= Tx * tf.sqrt(1/tf.to_float(Tx))

        # Restore shape for residual connection
        tensor = fc_block(ctx, hp.embed_size, training=training)

        # returns the alignment of the first one
        alignments = tf.transpose(alignments[0])[::-1, :]

    return tensor, alignments, max_attentions

def attention(inputs, attention_size):
    """
    Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector.
    The idea was proposed in the article by Z. Yang et al., "Hierarchical Attention Networks
     for Document Classification", 2016: http://www.aclweb.org/anthology/N16-1174.
    Variables notation is also inherited from the article
    
    Args:
        inputs: The Attention inputs.
            Matches outputs of RNN/Bi-RNN layer (not final state):
                In case of RNN, this must be RNN outputs `Tensor`:
                    If time_major == False (default), this must be a tensor of shape:
                        `[batch_size, max_time, cell.output_size]`.
                    If time_major == True, this must be a tensor of shape:
                        `[max_time, batch_size, cell.output_size]`.
                In case of Bidirectional RNN, this must be a tuple (outputs_fw, outputs_bw) containing the forward and
                the backward RNN outputs `Tensor`.
                    If time_major == False (default),
                        outputs_fw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_fw.output_size]`
                        and outputs_bw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_bw.output_size]`.
                    If time_major == True,
                        outputs_fw is a `Tensor` shaped:
                        `[max_time, batch_size, cell_fw.output_size]`
                        and outputs_bw is a `Tensor` shaped:
                        `[max_time, batch_size, cell_bw.output_size]`.
        attention_size: Linear size of the Attention weights.
        time_major: The shape format of the `inputs` Tensors.
            If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
            If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
            Using `time_major = True` is a bit more efficient because it avoids
            transposes at the beginning and end of the RNN calculation.  However,
            most TensorFlow data is batch-major, so by default this function
            accepts input and emits output in batch-major form.
        return_alphas: Whether to return attention coefficients variable along with layer's output.
            Used for visualization purpose.
    Returns:
        The Attention output `Tensor`.
        In case of RNN, this will be a `Tensor` shaped:
            `[batch_size, cell.output_size]`.
        In case of Bidirectional RNN, this will be a `Tensor` shaped:
            `[batch_size, cell_fw.output_size + cell_bw.output_size]`.
    """

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

    # Trainable parameters
    #W_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    #b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    #u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    W_omega = tf.get_variable('W_omega',dtype=tf.float32, initializer=tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.get_variable('b_omega',dtype=tf.float32, initializer=tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.get_variable('u_omega',dtype=tf.float32, initializer=tf.random_normal([attention_size], stddev=0.1))

    # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
    #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size  
    v = tf.tanh(tf.tensordot(inputs, W_omega, axes=1) + b_omega) # [16,200,A]

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1)   # (B,T) shape
    alphas = tf.nn.softmax(vu)              # (B,T) shape also [16,200]
    alphas = tf.reshape(alphas,inputs.shape[0:2])

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    return output