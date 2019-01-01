import os, pickle
from tensorflow.examples.tutorials.mnist import input_data
from datetime import  datetime

from utils import *
import ops

'''
    For cifar-10 dataset only
'''

class CWGAN_GP(object):
    model_name = 'condition_wgan_gp'

    def __init__(self, input_shape, batch_size=32, z_dim=128, g_filter_num=64, d_filter_num=64,
                 g_fc_dim=1024, d_fc_dim=1024, learning_rate=0.0001, gp_lambda=10.0, beta1=0.5,
                 model_path='./wgan_gp/checkpoint/', log_dir='.\\wgan_gp\\logs', pre_train=True):
        # copy parameters
        # input_shape should be 3 dimensional list like [height, width, channels]
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.g_filter_num = g_filter_num
        self.d_filter_num = d_filter_num
        self.g_fc_dim = g_fc_dim
        self.d_fc_dim = d_fc_dim
        self.gp_lambda = gp_lambda
        self.model_path = model_path

        self._build_model()
        self.g_loss, self.d_loss = self._loss_function()
        self.d_optim = tf.train.AdamOptimizer(learning_rate, beta1, 0.9).minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(learning_rate, beta1, 0.9).minimize(self.g_loss, var_list=self.g_vars)

        self.saver = tf.train.Saver()
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        # Initialize summary
        self.d_loss_real_summary = tf.summary.scalar('d_loss_real', self.d_loss_real)
        self.d_loss_fake_summary = tf.summary.scalar('d_loss_fake', self.d_loss_fake)
        self.gradient_penalty_summary = tf.summary.scalar('gradient_penalty', self.gradient_penalty)
        self.g_loss_summary = tf.summary.scalar('g_loss', self.g_loss)
        self.d_loss_summary = tf.summary.scalar('d_loss', self.d_loss)
        self.d_summary = tf.summary.merge([self.d_loss_real_summary, self.d_loss_summary, self.gradient_penalty_summary])
        self.g_summary = tf.summary.merge([self.d_loss_fake_summary, self.g_loss_summary])
        self.writer = tf.summary.FileWriter(log_dir, self.sess.graph)

        # Load pre-trained model if checkpoint is not empty
        if pre_train and len(os.listdir(self.model_path)) != 0:
            _, self.counter = self.load()
        else:
            print('Build model from scratch!!')
            self.counter = 0

    def _build_model(self):
        # Initialize batch normalizations (only for Generator)
        self.g_bn0 = ops.batch_norm(name='g_bn0')
        self.g_bn1 = ops.batch_norm(name='g_bn1')
        self.g_bn2 = ops.batch_norm(name='g_bn2')
        self.g_bn3 = ops.batch_norm(name='g_bn3')

        # Define inputs as tf.placeholder
        self.z_latent = tf.placeholder(tf.float32, shape=(None, self.z_dim), name='z_latent')
        self.real_images_vector = tf.placeholder(tf.float32, shape=[self.batch_size, None])
        self.real_images = tf.reshape(self.real_images_vector, shape=[self.batch_size]+self.input_shape)

        # D is the output of real images while D_ is generated output
        self.Generator = self._build_generator(is_training=True, reuse=False)
        self.Sampler = self._build_generator(is_training=False, reuse=True)
        self.Discriminator = self._build_discriminator(self.real_images, reuse=False)
        self.Discriminator_ = self._build_discriminator(self.Generator, reuse=True)

        trainable_vars = tf.trainable_variables()
        self.d_vars = [var for var in trainable_vars if 'Discriminator' in var.name]
        self.g_vars = [var for var in trainable_vars if 'Generator' in var.name]

    def _build_discriminator(self, input_op, reuse=False):
        with tf.variable_scope('Discriminator', reuse=reuse) as scope:
            self.conv_1 = ops.conv2d(input_op, n_out=self.d_filter_num, name='conv_1', activate='lrelu')

            # Convolutional layers
            conv_2 = ops.conv2d(self.conv_1, n_out=self.d_filter_num * 2, name='conv_2', activate='linear')
            self.conv_2 = ops.leaky_relu(conv_2)
            conv_3 = ops.conv2d(self.conv_2, n_out=self.d_filter_num * 4, name='conv_3', activate='linear')
            self.conv_3 = ops.leaky_relu(conv_3)
            conv_4 = ops.conv2d(self.conv_3, n_out=self.d_filter_num * 8, name='conv_4', activate='linear')
            self.conv_4 = ops.leaky_relu(conv_4)

            # Remove sigmoid in last layer
            d_reshaped = tf.reshape(self.conv_4, shape=[self.batch_size, -1])
            d_mapping = ops.fully_connect(d_reshaped, n_out=1, name='d_mapping')
            return d_mapping

    def _build_generator(self, is_training=True, reuse=False):
        with tf.variable_scope('Generator', reuse=reuse):
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
            self.z_relu = tf.nn.relu(self.g_bn0(self.z_reshaped, is_training=is_training))

            deconv_1 = ops.deconv2d(self.z_relu,
                                    output_shape=[self.batch_size, s_h8, s_w8, self.g_filter_num * 4],
                                    name='deconv_1', activate='linear', with_kernels=False)
            self.deconv_1 = tf.nn.relu(self.g_bn1(deconv_1, is_training=is_training))
            deconv_2 = ops.deconv2d(self.deconv_1,
                                    output_shape=[self.batch_size, s_h4, s_w4, self.g_filter_num * 2],
                                    name='deconv_2', activate='linear', with_kernels=False)
            self.deconv_2 = tf.nn.relu(self.g_bn2(deconv_2, is_training=is_training))
            deconv_3 = ops.deconv2d(self.deconv_2,
                                    output_shape=[self.batch_size, s_h2, s_w2, self.g_filter_num * 1],
                                    name='deconv_3', activate='linear', with_kernels=False)
            self.deconv_3 = tf.nn.relu(self.g_bn3(deconv_3, is_training=is_training))
            deconv_4 = ops.deconv2d(self.deconv_3,
                                    output_shape=[self.batch_size, s_h, s_w, self.input_shape[-1]],
                                    name='deconv4', activate='linear', with_kernels=False)
            return tf.nn.tanh(deconv_4)

    def _loss_function(self):
        # The original GAN loss definition
        self.d_loss_real = tf.reduce_mean(tf.negative(self.Discriminator))
        self.d_loss_fake = tf.reduce_mean(self.Discriminator_)
        d_loss = self.d_loss_real + self.d_loss_fake
        g_loss = tf.negative(self.d_loss_fake)

        # Gradient penalty
        epsilon = tf.random_uniform(shape=self.real_images.get_shape(), minval=0.0, maxval=1.0)
        differences = self.Generator - self.real_images
        interpolates = self.real_images + (epsilon * differences)
        self.discriminator_gp = self._build_discriminator(interpolates, reuse=True)
        gradients = tf.gradients(self.discriminator_gp, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        self.gradient_penalty = tf.reduce_mean(tf.square(slopes - 1))
        # The higher value of gp_lambda, the more stable, but the slower convergence
        d_loss += self.gp_lambda * self.gradient_penalty
        return g_loss, d_loss

    def train(self, reader, loop=20000, print_iteration=100, save_iteration=500,
              critic_iters=1, auto_save_img=True, auto_save_model=False):
        # sample_z is used to save generated image within each n iterations
        sample_z = np.random.uniform(-1.0, 1.0, size=(self.batch_size, self.z_dim)).astype(np.float32)

        for it in range(loop):
            batch_z = np.random.uniform(-1.0, 1.0, size=(self.batch_size, self.z_dim)).astype(np.float32)
            batch_images, _ = reader.next_batch(self.batch_size)

            # Update discriminator ONCE
            for j in range(critic_iters):
                _, summary_str = self.sess.run([self.d_optim, self.d_summary],
                                               feed_dict={self.real_images_vector: batch_images,
                                                          self.z_latent: batch_z})
                self.writer.add_summary(summary_str, self.counter)

            # Update generator once
            _, summary_str = self.sess.run([self.g_optim, self.g_summary],
                                           feed_dict={self.z_latent: batch_z})
            self.writer.add_summary(summary_str, self.counter)

            # print loss on screen every 100 step
            if it % print_iteration == 1:
                d_loss_fake_tmp, d_loss_real_tmp, d_gp_tmp, g_loss_tmp = self.sess.run(
                    fetches=[self.d_loss_fake, self.d_loss_real, self.gradient_penalty, self.g_loss],
                    feed_dict={self.z_latent: batch_z, self.real_images_vector: batch_images}
                )
                logging = ' --Iteration %d --Discriminator Loss: %g --Generator Loss: %g --Gradient Penalty: %g' %\
                          (it, d_loss_real_tmp + d_loss_fake_tmp, g_loss_tmp, d_gp_tmp)
                print(str(datetime.now()) + logging)
            # Save Images every 500 iterations automatically
            if auto_save_img and it % save_iteration == 1:
                samples = self.sess.run(self.Sampler, feed_dict={self.z_latent: sample_z})
                save_image(samples, name='WGAN_GP.png', idx=self.counter, path='./wgan_gp/generated/')
            # Save model every 1000 iterations automatically
            if auto_save_model and it % (2*save_iteration) == (save_iteration + 2):
                self.save()
            self.counter += 1
        print(str(datetime.now()) + ' --Training Finished!')
        if 'y' in str(input('Save Model??')):
            self.save()
        print('Done!!')

    def save(self):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        elif len(os.listdir(self.model_path)) != 0:
            fs = os.listdir(self.model_path)
            for f in fs:
                os.remove(self.model_path + f)
        save_path = self.saver.save(self.sess, self.model_path + 'wgan_gp.model', global_step=self.counter)
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
    reader = cifar10Reader()

    gan = CWGAN_GP(input_shape=[32, 32, 3], pre_train=True)
    show_all_variables()
    gan.train(reader, loop=20000, critic_iters=5, save_iteration=1550, auto_save_img=True, auto_save_model=False)

    print('Done!!')