import tensorflow as tf
import numpy as np
import os, sys
from datetime import datetime

import utils
from utils import conv2d, pooling, lrelu, fully_connect, deconv2d, batch_norm


class Network(object):
    name = 'face_AmbientGAN.model'

    def __init__(self, input_shape, learning_rate=2e-4, batch_size=64, z_dim=128, gp_lambda=0.25,
                 model_dir='./face/checkpoint/', log_dir='./face/logs', img_dir='./face/generated/',
                 data_dir='/home/zcx/Documents/datasets/AnimeFaces/', epoch=25, pre_train=True):
        self.input_dim = input_shape[0] * input_shape[1] * input_shape[2]
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.epoch = epoch
        self.z_dim = z_dim
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.img_dir = img_dir
        if os.path.exists(img_dir):
            os.makedirs(img_dir)

        reader = self._init_reader(data_dir)
        self.batch_xs = reader.make_one_shot_iterator().get_next()
        self.org_xs = tf.reshape(self.batch_xs, shape=[-1] + input_shape, name='org_xs')
        self.lossy_xs = tf.nn.dropout(self.org_xs, keep_prob=0.4)
        self.batch_zs = self.lossy_xs

        self._build_model()
        t_vars = tf.trainable_variables()
        g_vars = [var for var in t_vars if 'Generator' in var.name]
        d_vars = [var for var in t_vars if 'Discriminator' in var.name]
        self.g_loss, self.d_loss = self._loss_function(gp_lambda)
        self.d_optim = tf.train.AdamOptimizer(learning_rate, 0.5, 0.9).minimize(self.d_loss, var_list=d_vars)
        self.g_optim = tf.train.AdamOptimizer(learning_rate, 0.5, 0.9).minimize(self.g_loss, var_list=g_vars)

        self.saver = tf.train.Saver()
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        # Initialize summary
        g_loss_summary = tf.summary.scalar('g_loss', self.g_loss)
        d_loss_summary = tf.summary.scalar('d_loss', self.d_loss)
        gp_summary = tf.summary.scalar('GP', self.gp)
        gen_xs_summary = tf.summary.image('generated_xs', self.gen_xs)
        gen_ys_summary = tf.summary.image('generated_ys', self.gen_ys)
        org_summary = tf.summary.image('org_xs', self.org_xs)
        lossy_summary = tf.summary.image('lossy_image', self.lossy_xs)
        self.loss_summary = tf.summary.merge([g_loss_summary, d_loss_summary, gp_summary])
        self.image_summary = tf.summary.merge([gen_xs_summary, gen_ys_summary, org_summary, lossy_summary])

        # Load pre-trained model if checkpoint is not empty
        if pre_train and len(os.listdir(self.model_dir)) != 0:
            _, self.counter = self.load()
        else:
            print('Build model from scratch!!')
            self.counter = 0

    def _init_reader(self, data_dir):
        def _parse_function(filename):
            x_img_str = tf.read_file(filename)
            x_img_decoded = tf.image.convert_image_dtype(tf.image.decode_jpeg(x_img_str), tf.float32)
            x_img_resized = tf.image.resize_images(x_img_decoded, size=[96, 96],
                                                   method=tf.image.ResizeMethod.BILINEAR)
            return x_img_resized

        x_files = os.listdir(data_dir)
        x_app = [os.path.join(data_dir, name) for name in x_files]
        data = tf.data.Dataset.from_tensor_slices((tf.constant(x_app)))
        data = data.map(_parse_function)
        return data.shuffle(buffer_size=1000).batch(self.batch_size).repeat(self.epoch)

    def _build_model(self):
        self.gen_xs, self.end_1 = self._build_generator(self.batch_zs, reuse=False, is_training=True)
        _, self.end_2 = self._build_generator(self.org_xs, reuse=True, is_training=False)
        for key in self.end_2.keys():
            self.end_2[key] = tf.stop_gradient(self.end_2[key])
        self.gen_ys = utils.random_mask(self.gen_xs)
        self.d_real = self._build_discriminator(self.lossy_xs, reuse=False)
        self.d_fake = self._build_discriminator(self.gen_ys, reuse=True)

    def _build_generator(self, input_op, reuse=False, is_training=True):
        """_build_generator
        UNet architecture definition
        """
        shape_2 = [self.batch_size, 12, 12, int(self.z_dim / 4)]
        shape_3 = [self.batch_size, 24, 24, int(self.z_dim / 8)]
        shape_4 = [self.batch_size, 48, 48, int(self.z_dim / 16)]
        shape_5 = [self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]]

        end_points = {}
        with tf.variable_scope('Generator', reuse=reuse):
            # Encoder
            conv_1 = conv2d(input_op, 64, name='conv_1', use_bias=False)
            conv_1 = tf.nn.relu(batch_norm(conv_1, 'bn_conv1', is_training=is_training))
            conv_2 = conv2d(conv_1, 128, name='conv_2', use_bias=False)
            conv_2 = tf.nn.relu(batch_norm(conv_2, 'bn_conv2', is_training=is_training))
            conv_3 = conv2d(conv_2, 256, name='conv_3', use_bias=False)
            conv_3 = tf.nn.relu(batch_norm(conv_3, 'bn_conv3', is_training=is_training))
            conv_4 = conv2d(conv_3, 512, name='conv_4', use_bias=False)
            conv_4 = tf.nn.relu(batch_norm(conv_4, 'bn_conv4', is_training=is_training))

            # Collect end points
            end_points['conv_1'] = conv_1
            end_points['conv_2'] = conv_2
            end_points['conv_3'] = conv_3
            end_points['conv_4'] = conv_4

            # Decoder
            deconv1 = deconv2d(conv_4, output_shape=shape_2, name='deconv1')
            bn1 = tf.nn.relu(batch_norm(deconv1, 'bn1', is_training=is_training))
            deconv2 = deconv2d(tf.concat([bn1, conv_4], axis=-1), output_shape=shape_3, name='deconv2')
            bn2 = tf.nn.relu(batch_norm(deconv2, 'bn2', is_training=is_training))
            deconv3 = deconv2d(tf.concat([bn2, conv_3], axis=-1), output_shape=shape_4, name='deconv3')
            bn3 = tf.nn.relu(batch_norm(deconv3, 'bn3', is_training=is_training))
            deconv4 = deconv2d(tf.concat([bn3, conv_2], axis=-1), output_shape=shape_5, name='deconv4')
            return tf.nn.sigmoid(deconv4, name='g_net'), end_points

    def _build_discriminator(self, input_op, reuse=False):
        # Re-implementation of DCGAN
        with tf.variable_scope('Discriminator', reuse=reuse):
            conv1 = lrelu(conv2d(input_op, n_out=64, dw=1, dh=1, name='conv1'))
            pool1 = pooling(conv1, name='pool1')
            conv2 = lrelu(conv2d(pool1, n_out=128, dw=1, dh=1, name='conv2'))
            pool2 = pooling(conv2, name='pool2')
            conv3 = lrelu(conv2d(pool2, n_out=256, dw=1, dh=1, name='conv3'))
            pool3 = pooling(conv3, name='pool3')
            conv4 = lrelu(conv2d(pool3, n_out=512, dw=1, dh=1, name='conv4'))
            pool4 = pooling(conv4, name='pool4')
            conv5 = lrelu(conv2d(pool4, n_out=512, dw=1, dh=1, name='conv5'))
            pool5 = pooling(conv5, name='pool5')
            reshaped = tf.reshape(pool5, shape=[-1, 512], name='reshaped')
            fc1 = lrelu(fully_connect(reshaped, n_out=128, name='fc1'))
            return fully_connect(fc1, n_out=1, name='d_net')

    def _loss_function(self, gp_lambda):
        # The loss of Generator
        d_real_loss = tf.reduce_mean(tf.negative(self.d_real))
        d_fake_loss = tf.reduce_mean(self.d_fake)
        g_loss = tf.negative(d_fake_loss)

        # Rec loss
        rec_loss = tf.reduce_mean(tf.squared_difference(self.gen_xs, self.org_xs))

        # Style loss
        style_loss = 0.0
        gamma = 0.75
        for key in self.end_1.keys():
            style_loss = style_loss + gamma * tf.abs(self.end_1[key] - self.end_2[key])
        style_loss = tf.reduce_mean(style_loss)

        # TODO: Adjust the coeffcients for better performance
        g_loss = g_loss + rec_loss + style_loss

        # Gradient penalty loss
        epsilon = tf.random_uniform(shape=self.gen_ys.get_shape(), minval=0.0, maxval=1.0)
        interpolates = self.gen_ys + epsilon * (self.gen_ys - self.lossy_xs)
        d_gp = self._build_discriminator(interpolates, reuse=True)
        grads = tf.gradients(d_gp, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grads), reduction_indices=[1]))
        self.gp = tf.reduce_mean(tf.square(slopes - 1), name='gradient_penalty')
        d_loss = d_real_loss + d_fake_loss + gp_lambda * self.gp
        return g_loss, d_loss

    def train(self, loop=20000, print_iter=50, save_iter=500):
        writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        for it in range(loop):
            try:
                self.sess.run(self.d_optim)
                self.sess.run(self.g_optim)
                if it % print_iter == 1:
                    dloss, gloss, summ = self.sess.run([self.d_loss, self.g_loss, self.loss_summary])
                    writer.add_summary(summ, self.counter)
                    words = ' --Iteration %d --Discriminator Loss: %g --Generator Loss: %g' %\
                            (self.counter, dloss, gloss)
                    print(str(datetime.now()) + words)
                if it % save_iter == 200:
                    summ = self.sess.run(self.image_summary)
                    writer.add_summary(summ, self.counter)
                    generated = self.sampling(self.batch_size)
                    utils.save_image(generated, name='ambient.png', idx=self.counter, path='./face/generated/')
                    print('Save %d generated images to local directory' % (self.batch_size))
                self.counter += 1
            except tf.errors.InvalidArgumentError:
                continue
            except tf.errors.OutOfRangeError:
                print('Epoch has Ended!! Ready to save...')
                self.save()
                return None
        print('Training finished!! Ready to save...')
        self.save()

    def sampling(self, num):
        ans = []
        for it in range(num):
            img = self.sess.run(self.gen_xs)
            ans.append(img)
        return ans

    def save(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        elif len(os.listdir(self.model_dir)) != 0:
            fs = os.listdir(self.model_dir)
            for f in fs:
                if 'ipynb' not in f:
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
    train_step = int(sys.argv[1])
    gan = Network(input_shape=[96, 96, 3], learning_rate=1e-5)
    utils.show_all_variables()

    gan.train(loop=train_step)

    print('\n', 'Done!!')
