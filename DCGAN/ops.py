import tensorflow as tf

class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name='batch_norm'):
        '''
        :param epsilon:
        :param momentum:
        :param name:
        '''
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, input_op, is_training=True):
        return tf.contrib.layers.batch_norm(input_op,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=is_training,
                                            scope=self.name)


def leaky_relu(input_op, leak=0.2, name='leaky_relu'):
    return tf.maximum(input_op, leak*input_op, name=name)

def conv2d(input_op, n_out, name, kh=5, kw=5, dh=2, dw=2, activate='lrelu'):
    # 1. 使用stride为2的卷积代替pooling层
    # 2. 这里初始化变量必须用tf.get_variable，方便后面共享变量
    # 3. 为防止梯度消失，激活函数用Leaky ReLU
    n_in = input_op.get_shape()[-1].value

    with tf.variable_scope(name):
        kernel = tf.get_variable(name='_w',
                                 shape=(kh, kw, n_in, n_out),
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())

        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
        biases = tf.get_variable(name='biases', shape=(n_out), initializer=tf.constant_initializer(0.0))
        z_out = tf.nn.bias_add(conv, biases)
        if activate == 'lrelu':
            activation = leaky_relu(z_out)
        elif activate == 'relu':
            activation = tf.nn.relu(z_out, name='relu')
        else:
            return z_out
        return activation

def conv_3x3_with_bn(input_op, n_out, name, batch_norm, kh=3, kw=3, dh=1, dw=1, activate='relu'):
    n_in = input_op.get_shape()[-1].value

    with tf.variable_scope(name):
        kernel = tf.get_variable(name='conv_kernel',
                                 shape=(kh, kw, n_in, n_out),
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.get_variable(name='biases', shape=(n_out), initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input_op, kernel, strides=(1,dh,dw,1), padding='SAME')
        z_out = tf.nn.bias_add(conv, biases)
        if activate == 'lrelu':
            activation = leaky_relu(batch_norm(z_out))
        elif activate == 'relu':
            activation = tf.nn.relu(batch_norm(z_out), name='relu')
        else:
            return batch_norm(z_out)
        return activation

def deconv2d(input_op, output_shape, kh=5, kw=5, dh=2, dw=2, name='deconv', activate='lrelu', with_kernels=False):
    n_in = input_op.get_shape()[-1].value
    n_out = output_shape[-1]

    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        kernel = tf.get_variable(name='kernels',
                                 shape=(kh, kw, n_out, n_in),
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())

        deconv = tf.nn.conv2d_transpose(input_op, kernel,
                                        output_shape=output_shape,
                                        strides=(1, dh, dw, 1))
        biases = tf.get_variable(name='biases', shape=(n_out), initializer=tf.constant_initializer(0.0))
        z_out = tf.nn.bias_add(deconv, biases)
        if activate == 'lrelu':
            activation = leaky_relu(z_out)
        elif activate == 'relu':
            activation = tf.nn.relu(z_out)
        else:
            activation = z_out
        if with_kernels:
            return activation, kernel, biases
        else:
            return activation

def fully_connect(input_op, n_out, name='fully_connected', bias_init=0.0, activate='lrelu', with_kernels=False):
    n_in = input_op.get_shape()[-1].value

    with tf.variable_scope(name) as scope:
        kernel = tf.get_variable(name='matrix',
                                 shape=[n_in, n_out],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())

        biases = tf.get_variable(name='bias', shape=(n_out), initializer=tf.constant_initializer(bias_init))
        z_out = tf.matmul(input_op, kernel) + biases
        if activate == 'lrelu':
            activation = leaky_relu(z_out)
        elif activate == 'relu':
            activation = tf.nn.relu(z_out, name=scope)
        else:
            activation = z_out
        if with_kernels:
            return activation, kernel, biases
        else:
            return activation