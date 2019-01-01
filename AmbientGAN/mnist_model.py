import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os, sys
from datetime import datetime

import utils
from utils import conv2d, pooling, lrelu, fully_connect, deconv2d, batch_norm


class Network(object):
    name = 'MNIST_AmbientGAN.model'

    def __init__(self, input_shape, learning_rate=8e-4, keep_prob=0.25, batch_size=64, z_dim=128,
                 model_dir='./mnist/checkpoint/', log_dir='./mnist/logs', img_dir='./mnist/generated/',
                 pre_train=True):
        self.input_dim = input_shape[0] * input_shape[1] * input_shape[2]
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.log_dir = log_dir
        self.z_dim = z_dim
        self.keep_prob = keep_prob
        self.model_dir = model_dir
        self.img_dir = img_dir

        self.reader = input_data.read_data_sets('./MNIST_data/', one_hot=True)
        self.batch_xs = tf.placeholder(tf.float32, shape=[None, self.input_dim], name='batch_xs')
        self.batch_zs = tf.random_uniform((self.batch_size, z_dim), -1.0, 1.0, dtype=tf.float32)
        self.org_xs = tf.reshape(self.batch_xs, shape=[-1] + input_shape, name='org_xs')
        self.lossy_xs = tf.nn.dropout(self.org_xs, keep_prob=keep_prob, name='lossy_xs')

        self._build_model()
        t_vars = tf.trainable_variables()
        g_vars = [var for var in t_vars if 'Generator' in var.name]
        d_vars = [var for var in t_vars if 'Discriminator' in var.name]
        self.g_loss, self.d_loss = self._loss_function()
        self.d_optim = tf.train.AdamOptimizer(learning_rate).minimize(self.d_loss, var_list=d_vars)
        self.g_optim = tf.train.AdamOptimizer(learning_rate).minimize(self.g_loss, var_list=g_vars)

        self.saver = tf.train.Saver()
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        # Initialize summary
        g_loss_summary = tf.summary.scalar('g_loss', self.g_loss)
        d_loss_summary = tf.summary.scalar('d_loss', self.d_loss)
        gen_xs_summary = tf.summary.image('generated_xs', self.gen_xs)
        gen_ys_summary = tf.summary.image('generated_ys', self.gen_ys)
        org_summary = tf.summary.image('org_xs', self.org_xs)
        lossy_summary = tf.summary.image('lossy_image', self.lossy_xs)
        self.loss_summary = tf.summary.merge([g_loss_summary, d_loss_summary])
        self.image_summary = tf.summary.merge([gen_xs_summary, gen_ys_summary, org_summary, lossy_summary])

        # Load pre-trained model if checkpoint is not empty
        if pre_train and len(os.listdir(self.model_dir)) != 0:
            _, self.counter = self.load()
        else:
            print('Build model from scratch!!')
            self.counter = 0

    def _build_model(self):
        self.sample_zs = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='sample_zs')
        self.sampler = self._build_generator(self.sample_zs, reuse=False, is_training=False)
        self.gen_xs = self._build_generator(self.batch_zs, reuse=True, is_training=True)
        self.gen_ys = tf.nn.dropout(self.gen_xs, keep_prob=self.keep_prob, name='gen_ys')
        self.d_real, self.real_logits = self._build_discriminator(self.lossy_xs, reuse=False)
        self.d_fake, self.fake_logits = self._build_discriminator(self.gen_ys, reuse=True)

    def _build_generator(self, input_op, reuse=False, is_training=True):
        shape_1 = [self.batch_size, 2, 2, self.z_dim]
        shape_2 = [self.batch_size, 4, 4, int(self.z_dim / 2)]
        shape_3 = [self.batch_size, 7, 7, int(self.z_dim / 4)]
        shape_4 = [self.batch_size, 14, 14, int(self.z_dim / 8)]
        shape_5 = [self.batch_size, 28, 28, 1]

        with tf.variable_scope('Generator', reuse=reuse):
            z_mapping = tf.nn.relu(fully_connect(input_op, n_out=4*self.z_dim, name='z_map'))
            reshaped = tf.reshape(z_mapping, shape=shape_1, name='reshaped')
            deconv1 = deconv2d(reshaped, output_shape=shape_2, name='deconv1')
            bn1 = tf.nn.relu(batch_norm(deconv1, 'bn1', is_training=is_training))
            deconv2 = deconv2d(bn1, output_shape=shape_3, name='deconv2')
            bn2 = tf.nn.relu(batch_norm(deconv2, 'bn2', is_training=is_training))
            deconv3 = deconv2d(bn2, output_shape=shape_4, name='deconv3')
            bn3 = tf.nn.relu(batch_norm(deconv3, 'bn3', is_training=is_training))
            deconv4 = deconv2d(bn3, output_shape=shape_5, name='deconv4')
            return tf.nn.sigmoid(deconv4, name='g_net')

    def _build_discriminator(self, input_op, reuse=False):
        # Re-implementation of DCGAN
        with tf.variable_scope('Discriminator', reuse=reuse):
            conv1 = lrelu(batch_norm(conv2d(input_op, n_out=64, name='conv1'), 'bn1', is_training=True))
            conv2 = lrelu(batch_norm(conv2d(conv1, n_out=128, name='conv2'), 'bn2', is_training=True))
            conv3 = lrelu(batch_norm(conv2d(conv2, n_out=256, name='conv3'), 'bn3', is_training=True))
            conv4 = lrelu(batch_norm(conv2d(conv3, n_out=512, name='conv4'), 'bn4', is_training=True))
            reshaped = tf.reshape(conv4, shape=[-1, 512], name='reshaped')
            fc1 = lrelu(fully_connect(reshaped, n_out=1, name='fc1'))
            return tf.nn.sigmoid(fc1, name='d_net'), fc1

    def _loss_function(self):
        d_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.real_logits,
                                                                             labels=tf.ones_like(self.d_real)))
        d_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logits,
                                                                             labels=tf.zeros_like(self.d_fake)))
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logits,
                                                                        labels=tf.ones_like(self.d_fake)))
        d_loss = d_real_loss + d_fake_loss
        return g_loss, d_loss

    def train(self, loop=20000, print_iter=50, save_iter=500):
        writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        batch_z = np.random.uniform(-1.0, 1.0, size=(self.batch_size, self.z_dim))
        for it in range(loop):
            xss, _ = self.reader.train.next_batch(self.batch_size)
            self.sess.run(self.d_optim, feed_dict={self.batch_xs: xss})
            # Update G twice to avoid gradient vanishing
            self.sess.run(self.g_optim, feed_dict={self.batch_xs: xss})
            self.sess.run(self.g_optim, feed_dict={self.batch_xs: xss})
            if it % print_iter == 1:
                dloss, gloss, summ = self.sess.run([self.d_loss, self.g_loss, self.loss_summary],
                                                   feed_dict={self.batch_xs: xss})
                writer.add_summary(summ, self.counter)
                words = ' --Iteration %d --Discriminator Loss: %g --Generator Loss: %g' %\
                        (self.counter, dloss, gloss)
                print(str(datetime.now()) + words)
            if it % save_iter == 200:
                summ = self.sess.run(self.image_summary, feed_dict={self.batch_xs: xss})
                writer.add_summary(summ, self.counter)
                generated = self.sampling(self.batch_size, batch_z=batch_z)
                utils.save_image(generated, name='ambient.png', idx=self.counter)
                print('Save %d generated images to local directory' % (self.batch_size))
            self.counter += 1
        print('Training finished!! Ready to save...')
        self.save()

    def sampling(self, num, batch_z=None):
        if batch_z is None:
            batch_z = np.random.uniform(-1.0, 1.0, size=(self.batch_size, self.z_dim))
        return self.sess.run(self.sampler, feed_dict={self.sample_zs: batch_z})

    def save(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        elif len(os.listdir(self.model_dir)) != 0:
            fs = os.listdir(self.model_dir)
            for f in fs:
                os.remove(self.model_dir + f)
        save_path = self.saver.save(self.sess, self.model_dir + self.name, global_step=self.counter)
        print('MODEL RESTORED IN: ' + save_path)

    def load(self):
        import re
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, self.model_dir + ckpt_name)
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0


if __name__ == '__main__':
    gan = Network(input_shape=[28, 28, 1])
    utils.show_all_variables()

    gan.train(2000)

    print('\n', 'Done!!')
