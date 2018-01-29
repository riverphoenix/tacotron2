# -*- coding: utf-8 -*-
#/usr/bin/python2


from __future__ import print_function

from tqdm import tqdm
import argparse
import os
from glob import glob

from data_load import get_batch, load_vocab
from hyperparams import Hyperparams as hp
from modules import *
from networks import encoder, decoder, converter
import tensorflow as tf
from fn_utils import infolog
import synthesize
from utils import *
from tensorflow.python import debug as tf_debug
from google.protobuf import text_format
from attention_wrapper import AttentionWrapper


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Graph:
    def __init__(self, config=None, training=True, train_form='Both'):
        # Load vocabulary
        self.char2idx, self.idx2char = load_vocab()
        self.graph = tf.Graph()
        with self.graph.as_default():
            if training:
                self.origx, self.x, self.y1, self.y2, self.y3, self.num_batch = get_batch(config,train_form)

            else: # Evaluation
                self.x = tf.placeholder(tf.int32, shape=(hp.batch_size, hp.T_x))
                self.y1 = tf.placeholder(tf.float32, shape=(hp.batch_size, hp.T_y//hp.r, hp.n_mels*hp.r))

            # Networks
            if train_form != 'Converter':
                with tf.variable_scope("encoder"):
                    self.encoded = encoder(self.x, training=training)
                    
                with tf.variable_scope("decoder"):
                    self.mel_output, self.bef_mel_output, self.done_output, self.decoder_state, self.LTSM, self.step = decoder(self.y1, self.encoded, training=training)
                    self.cell_state = self.decoder_state.cell_state
                    self.mel_output = tf.nn.sigmoid(self.mel_output)
                    # self.bef_mel_output = tf.nn.sigmoid(self.bef_mel_output)
                
            if train_form == 'Both':
                with tf.variable_scope("converter"):
                    self.converter_input = self.mel_output
                    self.mag_logits = converter(self.converter_input, training=training)
                    self.mag_output = tf.nn.sigmoid(self.mag_logits)
            elif train_form == 'Converter':
                with tf.variable_scope("converter"):
                    self.converter_input = self.y1
                    self.mag_logits = converter(self.converter_input, training=training)
                    self.mag_output = tf.nn.sigmoid(self.mag_logits)

            
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

            if training:
                # Loss
                if train_form != 'Converter':
                    self.loss1 = tf.reduce_mean(tf.abs(self.mel_output - self.y1))
                    self.loss1b = tf.reduce_mean(tf.abs(self.bef_mel_output - self.y1))
                    if hp.include_dones:
                        self.loss2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.done_output, labels=self.y2))
                if train_form != 'Encoder':
                    self.loss3= tf.reduce_mean(tf.abs(self.mag_output - self.y3))

                if train_form == 'Both':
                    if hp.include_dones:
                        self.loss = self.loss1 + self.loss1b + self.loss2 + self.loss3
                    else:
                        self.loss = self.loss1 + self.loss1b + self.loss3
                elif train_form == 'Encoder':
                    if hp.include_dones:
                        self.loss = self.loss1 + self.loss1b + self.loss2
                    else:
                        self.loss = self.loss1 + self.loss1b
                else:
                    self.loss = self.loss3
                
                # Training Scheme
                self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr)
                ## gradient clipping
                self.gvs = self.optimizer.compute_gradients(self.loss)
                self.clipped = []
                for grad, var in self.gvs:
                    grad = grad if grad is None else tf.clip_by_value(grad, -1. * hp.max_grad_val, hp.max_grad_val)
                    grad = grad if grad is None else tf.clip_by_norm(grad, hp.max_grad_norm)
                    self.clipped.append((grad, var))

                self.train_op = self.optimizer.apply_gradients(self.clipped, global_step=self.global_step)
                
                # Summary
                tf.summary.scalar('loss', self.loss)

                if train_form != 'Converter':
                    tf.summary.histogram('mel_output', self.mel_output)
                    tf.summary.histogram('mel_actual', self.y1)
                    tf.summary.scalar('loss1', self.loss1)
                    if hp.include_dones:
                        tf.summary.histogram('done_output', self.done_output)
                        tf.summary.histogram('done_actual', self.y2)
                        tf.summary.scalar('loss2', self.loss2)
                if train_form != 'Encoder':
                    tf.summary.histogram('mag_output', self.mag_output)
                    tf.summary.histogram('mag_actual', self.y3)
                    tf.summary.scalar('loss3', self.loss3)
              
                self.merged = tf.summary.merge_all()

class GraphTest:
    def __init__(self, config=None):
        self.char2idx, self.idx2char = load_vocab()
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.origx, _, _, _, _, _ = get_batch(config,'Encoder')
  
def main():
    print()
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', default=hp.logdir)
    parser.add_argument('--log_name', default=hp.logname)
    parser.add_argument('--sample_dir', default=hp.sampledir)
    parser.add_argument('--data_paths', default=hp.data)
    parser.add_argument('--load_path', default=None)
    parser.add_argument('--load_converter', default=None)
    parser.add_argument('--deltree', default=False)

    parser.add_argument('--summary_interval', type=int, default=hp.summary_interval)
    parser.add_argument('--test_interval', type=int, default=hp.test_interval)
    parser.add_argument('--checkpoint_interval', type=int, default=hp.checkpoint_interval)
    parser.add_argument('--num_iterations', type=int, default=hp.num_iterations)

    parser.add_argument('--debug',type=bool,default=False)

    config = parser.parse_args()

    config.log_dir = config.log_dir + '/' + config.log_name
    if not os.path.exists(config.log_dir): 
        os.makedirs(config.log_dir)
    elif config.deltree:
        for the_file in os.listdir(config.log_dir):
            file_path = os.path.join(config.log_dir, the_file)
            os.unlink(file_path)
    log_path = os.path.join(config.log_dir+'/', 'train.log')
    infolog.init(log_path, "log")
    checkpoint_path = os.path.join(config.log_dir, 'model.ckpt')

    if (hp.test_only == 0):
        g = Graph(config=config, training=True, train_form=hp.train_form)
        print("Training Graph loaded")
    if hp.test_graph or (hp.test_only > 0):
        g2 = Graph(config=config, training=False, train_form=hp.train_form)
        print("Testing Graph loaded")
        if config.load_converter or (hp.test_only > 0):
            g_conv = Graph(config=config, training=False, train_form='Converter')
            print("Converter Graph loaded")
    if (hp.test_only == 0):
        with g.graph.as_default():
            sv = tf.train.Supervisor(logdir=config.log_dir)
            with sv.managed_session() as sess:

                #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
                if config.load_path:
                        # Restore from a checkpoint if the user requested it.
                        infolog.log('Resuming from checkpoint: %s ' % (tf.train.latest_checkpoint(config.log_dir)), slack=True)
                        sv.saver.restore(sess, tf.train.latest_checkpoint(config.log_dir))
                else:        
                    infolog.log('Starting new training', slack=True)

                summary_writer = tf.summary.FileWriter(config.log_dir, sess.graph)
                
                for epoch in range(1, 100000000):
                    if sv.should_stop(): break
                    losses = [0,0,0,0,0]
                    #for step in tqdm(range(1)):
                    for step in tqdm(range(g.num_batch)):
                    # for step in range(g.num_batch):
                        if hp.train_form == 'Both':
                            if hp.include_dones:
                                gs,merged,loss,loss1,loss1b, loss2,loss3,_ = sess.run([g.global_step,g.merged,g.loss,g.loss1,g.loss1b,g.loss2,g.loss3, g.train_op])
                                loss_one = [loss,loss1,loss1b, loss2,loss3]
                            else:
                                gs,merged,loss,loss1,loss1b, loss3,_ = sess.run([g.global_step,g.merged,g.loss,g.loss1,g.loss1b,g.loss3, g.train_op])
                                loss_one = [loss,loss1,loss1b, loss3,0]
                        elif hp.train_form == 'Encoder':
                            if hp.include_dones:
                                gs,merged,loss,loss1,loss1b, loss2,_ = sess.run([g.global_step,g.merged,g.loss,g.loss1,g.loss1b,g.loss2,g.train_op])
                                loss_one = [loss,loss1,loss1b, loss2,0]
                                # x1, encoded,LTSM,done,bef = sess.run([g.x, g.encoded,g.LTSM,g.done_output,g.bef_mel_output])
                                # print("@@@@@@@@@@@@@@@@@@")
                                # print(x1.shape)
                                # print(x1[0,:])
                                # print("---------------------")
                                # print(encoded.shape)
                                # print(encoded[0,:5,:5])
                                # print("---------------------")
                                # print(LTSM.shape)
                                # print(LTSM[0,:5,:5])
                                # print("---------------------")
                                # print(done.shape)
                                # print(done[0,::100,:])
                                # print("---------------------")
                                # print(bef.shape)
                                # print(bef[0,::160,::10])
                                # print("@@@@@@@@@@@@@@@@@@")
                            else:
                                gs,merged,loss, loss1, loss1b, _ = sess.run([g.global_step,g.merged,g.loss, g.loss1, g.loss1b, g.train_op])
                                loss_one = [loss,loss1,loss1b,0,0]
                                # x1, encoded,mel, bef,state = sess.run([g.x, g.encoded,g.mel_output,g.bef_mel_output,g.cell_state])
                                # print("@@@@@@@@@@@@@@@@@@")
                                # print(x1.shape)
                                # print(x1[0,:])
                                # print("---------------------")
                                # print(encoded.shape)
                                # print(encoded[0,:5,:5])
                                # print("---------------------")
                                # print(mel.shape)
                                # print(mel[0,::160,::10])
                                # print("---------------------")
                                # print(bef.shape)
                                # print(bef[0,::160,::10])
                                # print("---------------------")
                                # print(state)
                                # print("@@@@@@@@@@@@@@@@@@")
                        else:
                            gs,merged,loss,_ = sess.run([g.global_step,g.merged,g.loss, g.train_op])
                            loss_one = [loss,0,0,0,0]

                        # print("Step "+str(gs)+": "+str(loss_one))

                        losses = [x + y for x, y in zip(losses, loss_one)]

                    losses = [x / g.num_batch for x in losses]
                    print("###############################################################################")
                    if hp.train_form == 'Both':
                        if hp.include_dones:
                            infolog.log("Global Step %d (%04d): Loss = %.8f Loss1 = %.8f Loss1b = %.8f Loss2 = %.8f Loss3 = %.8f" %(epoch,gs,losses[0],losses[1],losses[2],losses[3],losses[4]))
                        else:
                            infolog.log("Global Step %d (%04d): Loss = %.8f Loss1 = %.8f Loss1b = %.8f Loss3 = %.8f" %(epoch,gs,losses[0],losses[1],losses[2],losses[3]))
                    elif hp.train_form == 'Encoder':
                        if hp.include_dones:
                            infolog.log("Global Step %d (%04d): Loss = %.8f Loss1 = %.8f Loss1b = %.8f Loss2 = %.8f" %(epoch,gs,losses[0],losses[1],losses[2],losses[3]))
                        else:
                            infolog.log("Global Step %d (%04d): Loss = %.8f Loss1 = %.8f Loss1b = %.8f" %(epoch,gs,losses[0],losses[1],losses[2]))
                    else:
                        infolog.log("Global Step %d (%04d): Loss = %.8f" %(epoch,gs,losses[0]))
                    print("###############################################################################")

                    if epoch % config.summary_interval == 0:
                        infolog.log('Saving summary')
                        summary_writer.add_summary(merged,gs)
                        if hp.train_form == 'Both':
                            if hp.include_dones:
                                origx, Kmel_out,Ky1,Kdone_out,Ky2,Kmag_out,Ky3 = sess.run([g.origx, g.mel_output,g.y1,g.done_output,g.y2,g.mag_output,g.y3])
                                plot_losses(config,Kmel_out,Ky1,Kdone_out,Ky2,Kmag_out,Ky3,gs)
                            else:
                                origx, Kmel_out,Ky1,Kmag_out,Ky3 = sess.run([g.origx, g.mel_output,g.y1,g.mag_output,g.y3])
                                plot_losses(config,Kmel_out,Ky1,None,None,Kmag_out,Ky3,gs)
                        elif hp.train_form == 'Encoder':
                            if hp.include_dones:
                                origx, Kmel_out,Ky1,Kdone_out,Ky2 = sess.run([g.origx, g.mel_output,g.y1,g.done_output,g.y2])
                                plot_losses(config,Kmel_out,Ky1,Kdone_out,Ky2,None,None,gs)
                            else:
                                origx, Kmel_out,Ky1 = sess.run([g.origx, g.mel_output,g.y1])
                                plot_losses(config,Kmel_out,Ky1,None,None,None,None,gs)
                        else:
                            origx, Kmag_out,Ky3 = sess.run([g.origx, g.mag_output,g.y3])
                            plot_losses(config,None,None,None,None,Kmag_out,Ky3,gs)


                    if epoch % config.checkpoint_interval == 0:
                        infolog.log('Saving checkpoint to: %s-%d' % (checkpoint_path, gs))
                        sv.saver.save(sess, checkpoint_path, global_step=gs)

                    if hp.test_graph and hp.train_form !='Converter':
                        if epoch % config.test_interval == 0:
                            infolog.log('Saving audio')
                            origx = sess.run([g.origx])
                            if not config.load_converter:
                                wavs = synthesize.synthesize_part(g2,config,gs,origx,None)
                            else:
                                wavs = synthesize.synthesize_part(g2,config,gs,origx,g_conv)
                            plot_wavs(config,wavs,gs)

                    # break
                    if gs > config.num_iterations: break
    else:
        infolog.log('Saving audio')
        gT = GraphTest(config=config)
        with gT.graph.as_default():
            svT = tf.train.Supervisor(logdir=config.log_dir)
            with svT.managed_session() as sessT:
                origx = sessT.run([gT.origx])
        if not config.load_converter:
            wavs = synthesize.synthesize_part(g2,config,0,origx,None)
        else:
            wavs = synthesize.synthesize_part(g2,config,0,origx,g_conv)
        plot_wavs(config,wavs,0)

    print("Done")

if __name__ == '__main__':
    main()