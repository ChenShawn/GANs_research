import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from datetime import  datetime
import numpy as np
import os, math

from utils import *
import ops

# For MNIST dataset
class DCGAN(object):
    model_name = 'dcgan'

    def __init__(self, input_shape, batch_size=32, z_dim=100,
                 g_filter_num=64, d_filter_num=64, g_fc_dim=1024, d_fc_dim=1024,
                 learning_rate=0.0002, beta1=0.5, model_path='./checkpoint/',
                 log_dir='.\\logs', pre_train=True):
        # copy parameters
        # input_shape should be 3 dimensional list
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.g_filter_num = g_filter_num
        self.d_filter_num = d_filter_num
        self.g_fc_dim = g_fc_dim
        self.d_fc_dim = d_fc_dim
        self.model_path = model_path

        self._build_model()
        self.g_loss, self.d_loss = self._loss_function()
        self.d_optim = tf.train.AdamOptimizer(learning_rate, beta1).minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(learning_rate, beta1).minimize(self.g_loss, var_list=self.g_vars)

        self.saver = tf.train.Saver()
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        # Initialize summary
        self.d_loss_real_summary = tf.summary.scalar('d_loss_real', self.d_loss_real)
        self.d_loss_fake_summary = tf.summary.scalar('d_loss_fake', self.d_loss_fake)
        self.g_loss_summary = tf.summary.scalar('g_loss', self.g_loss)
        self.d_loss_summary = tf.summary.scalar('d_loss', self.d_loss)
        self.d_summary = tf.summary.merge([self.d_loss_real_summary, self.d_loss_summary])
        self.g_summary = tf.summary.merge([self.d_loss_fake_summary, self.g_loss_summary])
        self.writer = tf.summary.FileWriter(log_dir, self.sess.graph)

        # Load pre-trained model if checkpoint is not empty
        if pre_train and len(os.listdir(self.model_path)) != 0:
            _, self.counter = self.load()
        else:
            print('Build model from scratch!!')
            self.counter = 0

    def _build_model(self):
        # Utilize batch normalization in ops.py
        self.d_bn0 = ops.batch_norm(name='d_bn0')
        self.d_bn1 = ops.batch_norm(name='d_bn1')
        self.d_bn2 = ops.batch_norm(name='d_bn2')
        self.g_bn0 = ops.batch_norm(name='g_bn0')
        self.g_bn1 = ops.batch_norm(name='g_bn1')
        self.g_bn2 = ops.batch_norm(name='g_bn2')
        self.g_bn3 = ops.batch_norm(name='g_bn3')

        # Define inputs as tf.placeholder
        self.z_latent = tf.placeholder(tf.float32, shape=(None, self.z_dim), name='z_latent')
        self.real_images_vector = tf.placeholder(dtype=tf.float32,
                                                 shape=[self.batch_size, None],
                                                 name='real_images_vector')
        self.real_images = tf.reshape(self.real_images_vector, shape=[self.batch_size] + self.input_shape)

        # reuse variable, D is the output of real images while D_ is generated output
        self.Generator = self._generator_build()
        self.Sampler = self._sampler_build()
        self.Discriminator, self.D_logits = self._discriminator_build(self.real_images, reuse=False)
        self.Discriminator_, self.D_logits_ = self._discriminator_build(self.Generator, reuse=True)

        # var_list for optimization
        trainable_vars = tf.trainable_variables()
        self.d_vars = [var for var in trainable_vars if 'Discriminator' in var.name]
        self.g_vars = [var for var in trainable_vars if 'Generator' in var.name]

    def _generator_build(self):
        with tf.variable_scope('Generator'):
            # The shape for deconv every iteration
            s_h, s_w = self.input_shape[0], self.input_shape[1]
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            # n_out is 8 times of filter number multiply deconv area
            self.z_mapping = ops.fully_connect(self.z_latent,
                                               n_out=s_h16*s_w16*8*self.g_filter_num,
                                               activate='linear', with_kernels=False)
            self.z_reshaped = tf.reshape(self.z_mapping, shape=[-1, s_h16, s_w16, self.g_filter_num * 8])
            self.z_relu = tf.nn.relu(self.g_bn0(self.z_reshaped))

            deconv_1 = ops.deconv2d(self.z_relu,
                                    output_shape=[self.batch_size, s_h8, s_w8, self.g_filter_num * 4],
                                    name='deconv_1', activate='linear', with_kernels=False)
            self.deconv_1 = tf.nn.relu(self.g_bn1(deconv_1))
            deconv_2 = ops.deconv2d(self.deconv_1,
                                    output_shape=[self.batch_size, s_h4, s_w4, self.g_filter_num * 2],
                                    name='deconv_2', activate='linear', with_kernels=False)
            self.deconv_2 = tf.nn.relu(self.g_bn2(deconv_2))
            deconv_3 = ops.deconv2d(self.deconv_2,
                                    output_shape=[self.batch_size, s_h2, s_w2, self.g_filter_num * 1],
                                    name='deconv_3', activate='linear', with_kernels=False)
            self.deconv_3 = tf.nn.relu(self.g_bn3(deconv_3))
            deconv_4 = ops.deconv2d(self.deconv_3,
                                    output_shape=[self.batch_size, s_h, s_w, self.input_shape[-1]],
                                    name='deconv4', activate='linear', with_kernels=False)
            return tf.nn.tanh(deconv_4)

    def _discriminator_build(self, input_op, reuse=False):
        with tf.variable_scope('Discriminator') as scope:
            if reuse:
                scope.reuse_variables()
            self.conv_1 = ops.conv2d(input_op, n_out=self.d_filter_num, name='conv_1', activate='lrelu')

            # Convolutional layers
            conv_2 = ops.conv2d(self.conv_1, n_out=self.d_filter_num * 2, name='conv_2', activate='linear')
            self.conv_2 = ops.leaky_relu(self.d_bn0(conv_2))
            conv_3 = ops.conv2d(self.conv_2, n_out=self.d_filter_num * 4, name='conv_3', activate='linear')
            self.conv_3 = ops.leaky_relu(self.d_bn1(conv_3))
            conv_4 = ops.conv2d(self.conv_3, n_out=self.d_filter_num * 8, name='conv_4', activate='linear')
            self.conv_4 = ops.leaky_relu(self.d_bn2(conv_4))

            # Use sigmoid in last layer
            d_reshaped = tf.reshape(self.conv_4, shape=[self.batch_size, -1])
            d_mapping = ops.fully_connect(d_reshaped, n_out=1, name='d_mapping')
            return tf.nn.sigmoid(d_mapping), d_mapping

    # The only difference between sampler and generator is that BN IS NOT TRAINABLE!
    def _sampler_build(self):
        with tf.variable_scope('Generator', reuse=True) as scope:
            s_h, s_w = self.input_shape[0], self.input_shape[1]
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            # n_out is 8 times of filter number multiply deconv area
            sampler_0 = ops.fully_connect(self.z_latent,
                                          n_out=s_h16*s_w16*8*self.g_filter_num,
                                          activate='linear', with_kernels=False)
            sampler_1 = tf.reshape(sampler_0, shape=[-1, s_h16, s_w16, self.g_filter_num * 8])
            sampler_2 = tf.nn.relu(self.g_bn0(sampler_1, is_training=False))

            sampler_3 = ops.deconv2d(sampler_2,
                                     output_shape=[self.batch_size, s_h8, s_w8, self.g_filter_num * 4],
                                     name='deconv_1', activate='linear', with_kernels=False)
            sampler_4 = tf.nn.relu(self.g_bn1(sampler_3, is_training=False))
            sampler_5 = ops.deconv2d(sampler_4,
                                     output_shape=[self.batch_size, s_h4, s_w4, self.g_filter_num * 2],
                                     name='deconv_2', activate='linear', with_kernels=False)
            sampler_6 = tf.nn.relu(self.g_bn2(sampler_5, is_training=False))
            sampler_7 = ops.deconv2d(sampler_6,
                                     output_shape=[self.batch_size, s_h2, s_w2, self.g_filter_num * 1],
                                     name='deconv_3', activate='linear', with_kernels=False)
            sampler_8 = tf.nn.relu(self.g_bn3(sampler_7, is_training=False))
            sampler_9 = ops.deconv2d(sampler_8,
                                     output_shape=[self.batch_size, s_h, s_w, self.input_shape[-1]],
                                     name='deconv4', activate='linear', with_kernels=False)
            return tf.nn.tanh(sampler_9)

    def train(self, reader, loop=20000, print_iteration=100, save_iteration=500, auto_save_img=True, auto_save_model=True):
        # sample_z is used to save generated image within each n iterations
        sample_z = np.random.uniform(-1.0, 1.0, size=(self.batch_size, self.z_dim)).astype(np.float32)

        for it in range(loop):
            batch_z = np.random.uniform(-1.0, 1.0, size=(self.batch_size, self.z_dim)).astype(np.float32)
            try:
                batch_images, _ = reader.next_batch(self.batch_size)
            except:
                batch_images = reader.next_batch(self.batch_size)

            # Update discriminator
            _, summary_str = self.sess.run([self.d_optim, self.d_summary],
                                           feed_dict={self.real_images_vector: batch_images,
                                                      self.z_latent: batch_z})
            self.writer.add_summary(summary_str, self.counter)

            # Update generator twice to avoid gradient vanishing
            _, summary_str = self.sess.run([self.g_optim, self.g_summary],
                                           feed_dict={self.z_latent: batch_z})
            self.writer.add_summary(summary_str, self.counter)
            _, summary_str = self.sess.run([self.g_optim, self.g_summary],
                                           feed_dict={self.z_latent: batch_z})
            self.writer.add_summary(summary_str, self.counter)

            # print loss on screen every 100 step
            if it % print_iteration == 1:
                d_loss_fake_tmp, d_loss_real_tmp, g_loss_tmp = self.sess.run(
                    fetches=[self.d_loss_fake, self.d_loss_real, self.g_loss],
                    feed_dict={self.z_latent: batch_z, self.real_images_vector: batch_images}
                )
                logging = ' --Iteration %d --Discriminator Loss: %g --Generator Loss: %g' %\
                          (it, d_loss_real_tmp + d_loss_fake_tmp, g_loss_tmp)
                print(str(datetime.now()) + logging)
            # Save Images every 500 iterations automatically
            if auto_save_img and it % save_iteration == 1:
                samples = self.sess.run(self.Sampler, feed_dict={self.z_latent: sample_z})
                save_image(samples, name='DCGAN.png', idx=self.counter)
            # Save model every 1000 iterations automatically
            if auto_save_model and it % (2*save_iteration) == (save_iteration + 2):
                self.save()
            self.counter += 1
        print(str(datetime.now()) + ' --Training Finished!')
        if 'y' in str(input('Save Model??')):
            self.save()
        print('Done!!')

    def _loss_function(self):
        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.Discriminator)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.Discriminator_)))

        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.Discriminator_)))
        d_loss = self.d_loss_real + self.d_loss_fake
        return g_loss, d_loss

    def save(self):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        elif len(os.listdir(self.model_path)) != 0:
            fs = os.listdir(self.model_path)
            for f in fs:
                os.remove(self.model_path + f)
        save_path = self.saver.save(self.sess, self.model_path + 'dcgan.model', global_step=self.counter)
        print('MODEL RESTORED IN: ' + save_path)

    def load(self):
        # self.saver.restore(self.sess, self.model_path + 'dcgan.ckpt')
        # print('LOAD FROM %s FINISHED' % (self.model_path + 'dcgan.ckpt'))
        import re
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self.model_path)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, self.model_path + ckpt_name)
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0


if __name__ == '__main__':
    # mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    reader = animeReader(path='E:\\datasets\\AnimeFaces\\', high=15000)

    dcgan = DCGAN(input_shape=[96, 96, 3], pre_train=False)
    show_all_variables()
    dcgan.train(reader, loop=10000, auto_save_img=True, auto_save_model=False)

    print('\n' + 'Done!!')