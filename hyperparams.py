# -*- coding: utf-8 -*-
#/usr/bin/python2

import math

def get_T_y(duration, sr, hop_length, r):
    '''Calculates number of paddings for reduction'''
    def _roundup(x):
        return math.ceil(x * .1) * 10
    T = _roundup(duration*sr/hop_length)
    num_paddings = r - (T % r) if T % r != 0 else 0
    T += num_paddings
    return T

class Hyperparams:
    '''Hyper parameters'''
    # signal processing
    sr = 22050 # Sampling rate.
    n_fft = 2048 # fft points (samples)
    frame_shift = 0.0125 # seconds 0.0125
    frame_length = 0.05 # seconds 0.05
    hop_length = int(sr*frame_shift) # samples  This is dependent on the frame_shift.
    win_length = int(sr*frame_length) # samples This is dependent on the frame_length.
    n_mels = 80 # Number of Mel banks to generate
    n_iter = 200 # Number of inversion iterations
    preemphasis = 0.97 # or None 0.97
    max_db = 100
    ref_db = 25
    lowcut = 125.0
    highcut = 7600.0
    dropout_rate = .5
    z_drop = .1
    norm_type = "ins" # TODO: weight normalization

    # Model
    r = 1 # Reduction factor 4
    outputs_per_step = 1
    n_feed = 200
    run_cmu = True
    sinusoid = False
    normalization  = True
    
    ## Enocder
    phon_drop = 0. #0.2
    if not run_cmu:
        vocab_size = 32
    else:
        vocab_size = 53
    embed_size = 512 # == e
    
    enc_layers = 3
    enc_kernel = 5
    enc_filters = 512
    enc_units = 256

    ## Decoder
    attention_size = 128
    att_parameter_init = 0.5
    dec_att_filters = 32
    dec_att_kernel = 31
    parallel_iterations = 1

    dec_layers = 4
    dec_filter_size = 5
    attention_win_size = 3

    dec_prenet_size = 128
    dec_LSTM_size = 512
    dec_LSTM_layers = 2
    dec_postnet_layers = 5
    dec_postnet_size = 5
    dec_postnet_filters = 512
    dec_projection = 512
    
    ## Converter
    converter_layers = 10
    converter_filter_size = 5
    converter_channels = 256
	
    # data
    max_duration = 10.0#10.10 # seconds
    T_x = 200 #200 # characters. maximum length of text.
    T_y = int(get_T_y(max_duration, sr, hop_length, r)) # Maximum length of sound (frames)

    # run options
    test_graph = False
    include_dones = False
    train_form = 'Converter' # 'Encoder', 'Converter', 'Both'
    test_only = 0
    print_shapes = True

    # training scheme
    optim = 'adam'
    lr = 0.001
    logdir = "logs"
    logname = 'demos'
    sampledir = 'samples'
    puresynth = 'logs/first2'
    batch_size = 16
    max_grad_norm = 100.
    max_grad_val = 5.
    num_iterations = 500000

    # Prepo params
    data = 'datasets/defaultS'
    prepro_gpu = 16
    
    # Training and Testing
    summary_interval = 1
    test_interval = 3
    checkpoint_interval = 1

    # Remove rnn_wrappers