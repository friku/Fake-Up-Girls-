from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import ops
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from functools import partial

conv = partial(slim.conv2d, activation_fn=None, weights_initializer=tf.truncated_normal_initializer(stddev=0.02))
dconv = partial(slim.conv2d_transpose, activation_fn=None, weights_initializer=tf.random_normal_initializer(stddev=0.02))
fc = partial(ops.flatten_fully_connected, activation_fn=None, weights_initializer=tf.random_normal_initializer(stddev=0.02))
relu = tf.nn.relu
lrelu = partial(ops.leak_relu, leak=0.2)
batch_norm = partial(slim.batch_norm, decay=0.9, scale=True, epsilon=1e-5, updates_collections=None)
ln = slim.layer_norm

def ChAug(y):
    y = tf.pad(y, [[0,0],[int(y.get_shape()[1])-1,int(y.get_shape()[1])-1],[int(y.get_shape()[2])-1,int(y.get_shape()[2])-1],[0,0]], "CONSTANT")
    w,h = 2,3
    for i in range():
        y[i] = tf.roll(y[i],shift=[w,h],axis=[1,2])
#    y = tf.pad(y, [[0,0],[y.shape[1]-1,y.shape[1]-1],[y.shape[2]-1,y.shape[2]-1],[0,0]], "CONSTANT")
    return y

def generator(z, dim=64, reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
    fc_bn_relu = partial(fc, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)

    with tf.variable_scope('generator', reuse=reuse):
        y = fc_bn_relu(z, 4 * 4 * dim * 8)
        y = tf.reshape(y, [-1, 4, 4, dim * 8])
        y = dconv_bn_relu(y, dim * 4, 5, 2)
        y = dconv_bn_relu(y, dim * 2, 5, 2)
        y = dconv_bn_relu(y, dim * 1, 5, 2)
        img = tf.tanh(dconv(y, 3, 5, 2))
        return img

def generator128(z, dim=64, reuse=True, training=True):
    with tf.variable_scope('generator', reuse=reuse):
        bn = partial(batch_norm, is_training=training)
        dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
        fc_bn_relu = partial(fc, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)

        y = fc_bn_relu(z, 4 * 4 * dim * 16)
        y = tf.reshape(y, [-1, 4, 4, dim * 16])
        y = dconv_bn_relu(y, dim * 8, 5, 2)
        y = dconv_bn_relu(y, dim * 4, 5, 2)
        y = dconv_bn_relu(y, dim * 2, 5, 2)
        y = dconv_bn_relu(y, dim * 1, 5, 2)
        img = tf.tanh(dconv(y, 3, 5, 2))
        return img

def generator256(z, dim=64, reuse=True, training=True):
    with tf.variable_scope('generator', reuse=reuse):
        bn = partial(batch_norm, is_training=training)
        dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
        fc_bn_relu = partial(fc, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)

        y = fc_bn_relu(z, 4 * 4 * dim * 32)
        y = tf.reshape(y, [-1, 4, 4, dim * 32])
        y = dconv_bn_relu(y, dim * 16, 5, 2)
        y = dconv_bn_relu(y, dim * 8, 5, 2)
        y = dconv_bn_relu(y, dim * 4, 5, 2)
        y = dconv_bn_relu(y, dim * 2, 5, 2)
        y = dconv_bn_relu(y, dim * 1, 5, 2)
        img = tf.tanh(dconv(y, 3, 5, 2))
        return img
    

def generator_ch(z,ch_input,ch_mask, dim=64, reuse=True, training=True):
    with tf.variable_scope('generator', reuse=reuse):
        bn = partial(batch_norm, is_training=training)
        dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
        fc_bn_relu = partial(fc, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
        y = fc_bn_relu(z, 4 * 4 * dim * 8)
        y = tf.reshape(y, [-1, 4, 4, dim * 8])
        
        y = tf.multiply(y,ch_mask)
        y = tf.add(y,ch_input)
        
        
        y = dconv_bn_relu(y, dim * 4, 5, 2)
        y = dconv_bn_relu(y, dim * 2, 5, 2)
        y = dconv_bn_relu(y, dim * 1, 5, 2)
        img = tf.tanh(dconv(y, 3, 5, 2))
        return img


def discriminator(img, dim=64, reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    conv_bn_lrelu = partial(conv, normalizer_fn=bn, activation_fn=lrelu, biases_initializer=None)

    with tf.variable_scope('discriminator', reuse=reuse):
        y = lrelu(conv(img, dim, 5, 2))
        y = conv_bn_lrelu(y, dim * 2, 5, 2)
        y = conv_bn_lrelu(y, dim * 4, 5, 2)
        y = conv_bn_lrelu(y, dim * 8, 5, 2)
        logit = fc(y, 1)
        return logit


def discriminator_wgan_gp(img, dim=64, reuse=True, training=True):
    with tf.variable_scope('discriminator', reuse=reuse):
        conv_ln_lrelu = partial(conv, normalizer_fn=ln, activation_fn=lrelu, biases_initializer=None)
        y = lrelu(conv(img, dim, 5, 2))
        y = conv_ln_lrelu(y, dim * 2, 5, 2)
        y = conv_ln_lrelu(y, dim * 4, 5, 2)
        y = conv_ln_lrelu(y, dim * 8, 5, 2)
        logit = fc(y, 1)
        return logit


def discriminator_wgan_gp_add(img, dim=64, reuse=True, training=True):
    conv_ln_lrelu = partial(conv, normalizer_fn=ln, activation_fn=lrelu, biases_initializer=None)

    with tf.variable_scope('discriminator', reuse=reuse):
        y = lrelu(conv(img, dim, 5, 2))
        y = conv_ln_lrelu(y, dim * 2, 5, 2)
        y = conv_ln_lrelu(y, dim * 4, 5, 2)
        y = conv_ln_lrelu(y, dim * 8, 5, 2)
        y = fc(y, 16)
        feature = y
        logit = fc(y, 1)
        return logit,feature

def discriminator_wgan_gp128(img, dim=64, reuse=True, training=True):
    with tf.variable_scope('discriminator', reuse=reuse):
        conv_ln_lrelu = partial(conv, normalizer_fn=ln, activation_fn=lrelu, biases_initializer=None)
        y = lrelu(conv(img, dim, 5, 2))
        y = conv_ln_lrelu(y, dim * 2, 5, 2)
        y = conv_ln_lrelu(y, dim * 4, 5, 2)
        y = conv_ln_lrelu(y, dim * 8, 5, 2)
        y = conv_ln_lrelu(y, dim * 16, 5, 2)
        logit = fc(y, 1)
        return logit

def discriminator_wgan_gp256(img, dim=64, reuse=True, training=True):
    with tf.variable_scope('discriminator', reuse=reuse):
        conv_ln_lrelu = partial(conv, normalizer_fn=ln, activation_fn=lrelu, biases_initializer=None)
        y = lrelu(conv(img, dim, 5, 2))
        y = conv_ln_lrelu(y, dim * 2, 5, 2)
        y = conv_ln_lrelu(y, dim * 4, 5, 2)
        y = conv_ln_lrelu(y, dim * 8, 5, 2)
        y = conv_ln_lrelu(y, dim * 16, 5, 2)
        y = conv_ln_lrelu(y, dim * 32, 5, 2)
        logit = fc(y, 1)
        return logit
##################################################################################
# Residual-block, Self-Attention-block
##################################################################################
def batch_n(x, is_training=True, scope='batch_norm'):
    return tf.layers.batch_normalization(x,
                                         momentum=0.9,
                                         epsilon=1e-05,
                                         training=is_training,
                                         name=scope)

def resblock_up(x_init, channels, use_bias=True, is_training=True, scope='resblock_up'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = batch_n(x_init)
            x = relu(x)
            x = tf.image.resize_nearest_neighbor(x, (2*x.shape[1],2*x.shape[2]))
            x = conv(x, channels, 3, 1)

        with tf.variable_scope('res2') :
            x = batch_n(x)
            x = relu(x)
            x = conv(x, channels, 3, 1)

        with tf.variable_scope('skip') :
            x_init = tf.image.resize_nearest_neighbor(x_init,(2*x.shape[1],2*x.shape[2]))
            x_init = conv(x_init, channels, 1, 1)

    return x + x_init

def resblock_down(x_init, channels, use_bias=True, is_training=True, scope='resblock_down'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = relu(x_init)
            x = conv(x, channels, 3,1)

        with tf.variable_scope('res2') :
            x = relu(x)
            x = conv(x, channels, 3,1)
            x = tf.layers.average_pooling2d(x,2,2)

        with tf.variable_scope('skip') :
            x_init = conv(x_init, channels, 1,1)
            x_init = tf.layers.average_pooling2d(x_init,2,2)

    return x + x_init


def hw_flatten(x) :
    return tf.reshape(x,[-1,x.shape[1]*x.shape[2], x.shape[-1]])

def self_attention_2(x, channels, scope='self_attention'):
    with tf.variable_scope(scope):
        f = conv(x, channels // 8, 1, 1, scope='f_conv')  # [bs, h, w, c']
        f = tf.layers.max_pooling2d(f, pool_size=2, strides=2, padding='SAME')

        g = conv(x, channels // 8, 1, 1, scope='g_conv')  # [bs, h, w, c']

        h = conv(x, channels // 2, 1, 1, scope='h_conv')  # [bs, h, w, c]
        h = tf.layers.max_pooling2d(h, pool_size=2, strides=2, padding='SAME')

        # N = h * w
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

        beta = tf.nn.softmax(s)  # attention map

        o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

        o = tf.reshape(o, shape=[-1, x.shape[1], x.shape[2], channels // 2])  # [bs, h, w, C]
        o = conv(o, channels, 1, 1, scope='attn_conv')
        x = gamma * o + x
    return x

def global_sum_pooling(x) :
    gsp = tf.reduce_sum(x, axis=[1, 2])
    return gsp


#GAN_model

def generator_big(z, dim=64, reuse=True, training=True):
    with tf.variable_scope('generator', reuse=reuse):
        bn = partial(batch_norm, is_training=training)
        fc_bn_relu = partial(fc, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
        y = fc_bn_relu(z, 4 * 4 * dim * 8)
        y = tf.reshape(y, [-1, 4, 4, dim * 8])
        y = resblock_up(y, dim * 4, scope='resblock_up_1')
        y = resblock_up(y, dim * 2, scope='resblock_up_2')
        y = resblock_up(y, dim * 1, scope='resblock_up_3')
        y = self_attention_2(y, dim * 1, scope='self_attention')
        y = resblock_up(y, 3, scope='resblock_up_4')
        img = tf.tanh(y)
        return img

def discriminator_wgan_gp_big(img, dim=64, reuse=True, training=True):
    with tf.variable_scope('discriminator', reuse=reuse):     
        y = resblock_down(img, dim * 1, scope='resblock_down_1')
        y = self_attention_2(y, dim * 1, scope='self_attention')
        y = resblock_down(y, dim * 2, scope='resblock_down_2')
        y = resblock_down(y, dim * 4, scope='resblock_down_3')
        y = resblock_down(y, dim * 8, scope='resblock_down_4')
        logit = fc(y, 1)
        return logit


def generator_self(z, dim=64, reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
    fc_bn_relu = partial(fc, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)

    with tf.variable_scope('generator', reuse=reuse):
        y = fc_bn_relu(z, 4 * 4 * dim * 8)
        y = tf.reshape(y, [-1, 4, 4, dim * 8])
        y = dconv_bn_relu(y, dim * 4, 3, 2)
        y = dconv_bn_relu(y, dim * 2, 3, 2)
        y = dconv_bn_relu(y, dim * 1, 3, 2)
        y = self_attention_2(y, dim * 1, scope='self_attention')
        img = tf.tanh(dconv(y, 3, 5, 2))
        return img


def discriminator_wgan_gp_self(img, dim=64, reuse=True, training=True):
    with tf.variable_scope('discriminator', reuse=reuse):
        conv_ln_lrelu = partial(conv, normalizer_fn=ln, activation_fn=lrelu, biases_initializer=None)
        y = lrelu(conv(img, dim, 5, 2))
        y = self_attention_2(y, dim * 1, scope='self_attention')
        y = conv_ln_lrelu(y, dim * 2, 3, 2)
        y = conv_ln_lrelu(y, dim * 4, 3, 2)
        y = conv_ln_lrelu(y, dim * 8, 3, 2)
        logit = fc(y, 1)
        return logit
