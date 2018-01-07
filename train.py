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


class Graph:
    def __init__(self, config=None,training=True):
        # Load vocabulary
        self.char2idx, self.idx2char = load_vocab()
        self.graph = tf.Graph()
        with self.graph.as_default():
            if training:
                self.origx, self.x, self.y1, self.y2, self.y3, self.num_batch = get_batch(config)

            else: # Evaluation
                self.x = tf.placeholder(tf.int32, shape=(1, hp.T_x))
                self.y1 = tf.placeholder(tf.float32, shape=(1, hp.T_y//hp.r, hp.n_mels*hp.r))

			# Get decoder inputs: feed last frames only
            self.decoder_input = tf.concat((tf.zeros_like(self.y1[:, :1, -hp.n_mels:]), self.y1[:, :-1, -hp.n_mels:]), 1)
            
            # Networks
            with tf.variable_scope("encoder"):
                self.encoded = encoder(self.x, training=training)
                
            with tf.variable_scope("decoder"):
                self.mel_logits, self.done_output = decoder(self.decoder_input, self.encoded, training=training)
                self.mel_output = self.mel_logits
                #self.mel_output = tf.nn.sigmoid(self.mel_logits)
                
            with tf.variable_scope("converter"):
                self.converter_input = tf.reshape(self.mel_output, (-1, hp.T_y, hp.n_mels))
                self.mag_logits = converter(self.converter_input, training=training)
                self.mag_output = tf.nn.sigmoid(self.mag_logits)
            
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

            if training:
                # Loss
                self.loss1 = tf.reduce_mean(tf.abs(self.mel_output - self.y1))
                self.loss2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.done_output, labels=self.y2))
                self.loss3= tf.reduce_mean(tf.abs(self.mag_output - self.y3))
                self.loss = self.loss1 + self.loss2 + self.loss3

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
                tf.summary.histogram('mel_output', self.mel_output)
                tf.summary.histogram('mel_actual', self.y1)
                tf.summary.histogram('done_output', self.done_output)
                tf.summary.histogram('done_actual', self.y2)
                tf.summary.histogram('mag_output', self.mag_output)
                tf.summary.histogram('mag_actual', self.y3)

                tf.summary.scalar('loss', self.loss)
                tf.summary.scalar('loss1', self.loss1)
                tf.summary.scalar('loss2', self.loss2)
                tf.summary.scalar('loss3', self.loss3)
              
                self.merged = tf.summary.merge_all()

def get_most_recent_checkpoint(checkpoint_dir):
    checkpoint_paths = [path for path in glob("{}/*.ckpt-*.data-*".format(checkpoint_dir))]
    idxes = [int(os.path.basename(path).split('-')[1].split('.')[0]) for path in checkpoint_paths]

    max_idx = max(idxes)
    lastest_checkpoint = os.path.join(checkpoint_dir, "model.ckpt-{}".format(max_idx))
    print(" [*] Found lastest checkpoint: {}".format(lastest_checkpoint))
    return lastest_checkpoint

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', default=hp.logdir)
    parser.add_argument('--log_name', default=hp.logname)
    parser.add_argument('--sample_dir', default=hp.sampledir)
    parser.add_argument('--data_paths', default=hp.data)
    parser.add_argument('--load_path', default=None)
    parser.add_argument('--initialize_path', default=None)
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
    g = Graph(config=config);
    print("Training Graph loaded")
    g2 = Graph(config=config,training=False);
    print("Testing Graph loaded")
    with g.graph.as_default():
        sv = tf.train.Supervisor(logdir=config.log_dir)
        with sv.managed_session() as sess:

            #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            if config.load_path:
                # Restore from a checkpoint if the user requested it.
                tf.reset_default_graph()
                restore_path = get_most_recent_checkpoint(config.load_path)
                sv.saver.restore(sess, restore_path)
                infolog.log('Resuming from checkpoint: %s ' % (restore_path), slack=True)
            elif config.initialize_path:
                restore_path = get_most_recent_checkpoint(config.initialize_path)
                sv.saver.restore(sess, restore_path)
                infolog.log('Initialized from checkpoint: %s ' % (restore_path), slack=True)
            else:
                infolog.log('Starting new training', slack=True)

            summary_writer = tf.summary.FileWriter(config.log_dir, sess.graph)
            
            for epoch in range(1, 100000000):
                if sv.should_stop(): break
                losses = [0,0,0,0]
                #losses = [0,0,0]
                for step in tqdm(range(g.num_batch)):
                #for step in range(g.num_batch):
                    #gs,merged,loss,loss1,loss3,alginm,_ = sess.run([g.global_step,g.merged,g.loss,g.loss1,g.loss3, g.alignments_li,g.train_op])
                    gs,merged,loss,loss1,loss2,loss3,alginm,_ = sess.run([g.global_step,g.merged,g.loss,g.loss1,g.loss2,g.loss3, g.alignments_li,g.train_op])
                    #infolog.log("Step %04d: Loss = %.8f Loss1 = %.8f Loss2 = %.8f Loss3 = %.8f" %(gs,loss,loss1,loss2,loss3))
                    #infolog.log("Step %04d: Loss = %.8f Loss1 = %.8f Loss3 = %.8f" %(gs,loss,loss1,loss3))
                    #loss_one = [loss,loss1,loss3]
                    loss_one = [loss,loss1,loss2,loss3]
                    losses = [x + y for x, y in zip(losses, loss_one)]

                losses = [x / g.num_batch for x in losses]
                print("###############################################################################")
                infolog.log("Global Step %d (%04d): Loss = %.8f Loss1 = %.8f Loss2 = %.8f Loss3 = %.8f" %(epoch,gs,losses[0],losses[1],losses[2],losses[3]))
                #infolog.log("Global Step %d (%04d): Loss = %.8f Loss1 = %.8f Loss3 = %.8f" %(epoch,gs,losses[0],losses[1],losses[2]))
                print("###############################################################################")

                if epoch % config.summary_interval == 0:
                    infolog.log('Saving summary')
                    summary_writer.add_summary(merged,gs)
                    origx, Kmel_out,Ky1,Kdone_out,Ky2,Kmag_out,Ky3 = sess.run([g.origx, g.mel_output,g.y1,g.done_output,g.y2,g.mag_output,g.y3])
                    #origx, Kmel_out,Ky1,Kmag_out,Ky3 = sess.run([g.origx, g.mel_output,g.y1,g.mag_output,g.y3])
                    plot_losses(config,Kmel_out,Ky1,Kdone_out,Ky2,Kmag_out,Ky3,gs)
                    #plot_losses(config,Kmel_out,Ky1,Kmag_out,Ky3,gs)


                if epoch % config.checkpoint_interval == 0:
                    infolog.log('Saving checkpoint to: %s-%d' % (checkpoint_path, gs))
                    sv.saver.save(sess, checkpoint_path, global_step=gs)

                if epoch % config.test_interval == 0:
                    infolog.log('Saving audio')
                    origx, Kmel_out,Ky1,Kdone_out,Ky2,Kmag_out,Ky3 = sess.run([g.origx, g.mel_output,g.y1,g.done_output,g.y2,g.mag_output,g.y3])
                    #origx, Kmel_out,Ky1,Kmag_out,Ky3 = sess.run([g.origx, g.mel_output,g.y1,g.mag_output,g.y3])
                    wavs = synthesize.synthesize_part(g2,config,gs,origx)
                    plot_wavs(config,wavs,gs)

                # break
                if gs > config.num_iterations: break

    print("Done")

if __name__ == '__main__':
    main()