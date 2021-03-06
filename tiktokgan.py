from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import glob
import utils
import traceback
import numpy as np
import tensorflow as tf
import models_64x64_pos as models
import horovod.tensorflow as hvd

""" param """
epoch = 50000
batch_i = 1
batch_size = 4*batch_i
lr_d = 0.0002
lr_g = 0.0002
z_dim = 100
n_critic = 2
gpu_id = 1
imgsize = 128

''' data '''
def preprocess_tik(img):
    sample_id = np.random.randint(0,img.shape[0],batch_size)
    batch = img[sample_id]
    batch = batch / 127.5 - 1
    return batch

tik = np.load("./tiktok_align_crop_all_resize128.npy")
hvd.init()
config = tf.ConfigProto()
config.gpu_options.visible_device_list = str(hvd.local_rank())
""" graphs """
# with tf.device(/gpu:%d' % gpu_id'):
  
''' models '''
generator = models.generator_self128
discriminator = models.discriminator_wgan_gp_self128
''' graph '''
# inputs
real = tf.placeholder(tf.float32, shape=[None, imgsize, imgsize, 3])
z = tf.placeholder(tf.float32, shape=[None, z_dim])

# generate
fake = generator(z, reuse=False)
# dicriminate
r_logit = discriminator(real, reuse=False)
f_logit = discriminator(fake)
# losses
def gradient_penalty(real, fake, f):
    def interpolate(a, b):
        shape = tf.concat((tf.shape(a)[0:1], tf.tile([1], [a.shape.ndims - 1])), axis=0)
        alpha = tf.random_uniform(shape=shape, minval=0., maxval=1.)
        inter = a + alpha * (b - a)
        inter.set_shape(a.get_shape().as_list())
        return inter
    x = interpolate(real, fake)
    pred = f(x)
    gradients = tf.gradients(pred, x)[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=list(range(1, x.shape.ndims))))
    gp = tf.reduce_mean((slopes - 1.)**2)
    return gp
wd = tf.reduce_mean(r_logit) - tf.reduce_mean(f_logit)
gp = gradient_penalty(real, fake, discriminator)
d_loss = -wd + gp * 10.0
g_loss = -tf.reduce_mean(f_logit)

d_opt = tf.train.AdamOptimizer(learning_rate=lr_d * hvd.size(), beta1=0.5)
d_opt = hvd.DistributedOptimizer(d_opt)
g_opt = tf.train.AdamOptimizer(learning_rate=lr_g * hvd.size(), beta1=0.5)
g_opt = hvd.DistributedOptimizer(g_opt)
hooks = [hvd.BroadcastGlobalVariablesHook(0)]
# otpims
d_var = utils.trainable_variables('discriminator')
g_var = utils.trainable_variables('generator')
d_step = d_opt.minimize(d_loss, var_list=d_var)
g_step = g_opt.minimize(g_loss, var_list=g_var)
# summaries
d_summary = utils.summary({wd: 'wd', gp: 'gp'})
g_summary = utils.summary({g_loss: 'g_loss'})
# sample
f_sample = generator(z, training=False)

""" train """
''' init '''
# session
sess = utils.session()
# iteration counter
it_cnt, update_cnt = utils.counter()
# saver
saver = tf.train.Saver(max_to_keep=5)
# summary writer

dir_name = "tik_"+str(imgsize)+"_lrd"+str(lr_d)+"_lrg"+str(lr_g)+"_ch"+str(lr_g)

summary_writer = tf.summary.FileWriter('./summaries/' + dir_name, sess.graph)

''' initialization '''
load_dir = './checkpoints/' + dir_name
ckpt_dir = './checkpoints/' + dir_name
utils.mkdir(ckpt_dir + '/')
if not utils.load_checkpoint(load_dir, sess):
    sess.run(tf.global_variables_initializer())
''' train '''
try:
    z_ipt_sample = np.random.normal(size=[25, z_dim])
    
    batch_epoch = 40000 // (batch_size * n_critic)
    max_it = epoch * batch_epoch
    for it in range(sess.run(it_cnt), max_it):
        sess.run(update_cnt)

        # which epoch
        epoch = it // batch_epoch
        it_epoch = it % batch_epoch + 1

        # train D
        for i in range(n_critic):
            real_ipt = preprocess_tik(tik)
            z_ipt = np.random.normal(size=[batch_size, z_dim])
            d_summary_opt, _ = sess.run([d_summary, d_step], feed_dict={real: real_ipt, z: z_ipt })
        summary_writer.add_summary(d_summary_opt, it)

        # train G
        z_ipt = np.random.normal(size=[batch_size, z_dim])
        g_summary_opt, _ = sess.run([g_summary, g_step], feed_dict={z: z_ipt })
        summary_writer.add_summary(g_summary_opt, it)

        # display
        if it % 1 == 0:
            print("Epoch: (%3d) (%5d/%5d)" % (epoch, it_epoch, batch_epoch))

        # save
        if (it + 1) % 1000 == 0:
            save_path = saver.save(sess, '%s/Epoch_(%d)_(%dof%d).ckpt' % (ckpt_dir, epoch, it_epoch, batch_epoch))
            print('Model saved in file: % s' % save_path)

        # sample
        if (it + 1) % 100 == 0:
            f_sample_opt = sess.run(f_sample, feed_dict={z: z_ipt_sample })

            save_dir = './sample_images_while_training/' + dir_name
            utils.mkdir(save_dir + '/')
            utils.imwrite(utils.immerge(f_sample_opt, 5, 5), '%s/Epoch_(%d)_(%dof%d).png' % (save_dir, epoch, it_epoch, batch_epoch))

except Exception:
    traceback.print_exc()
finally:
    print(" [*] Close main session!")
    sess.close()
