import numpy as np
import functools
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def doublewrap(function):
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator

@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator

class Model:
    def __init__(self, input_real, input_z, input_size, z_size, g_hidden_size, d_hidden_size, learning_rate, alpha, smooth):
        self.input_real = input_real
        self.input_z = input_z
        self.input_size = input_size
        self.z_size = z_size
        self.g_hidden_size = g_hidden_size
        self.d_hidden_size = d_hidden_size
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.smooth = smooth
        # self.generator
        # self.discriminator
        self.losses
        self.d_opt
        self.g_opt

    # @define_scope
    def generator(self, z, out_dim, n_units=128, reuse=False):
        with  tf.variable_scope('generator', reuse=reuse):
            h1 = tf.layers.dense(z, n_units, activation=None)
            h1 = tf.maximum(self.alpha*h1, h1)
            logits = tf.layers.dense(h1, out_dim, activation=None)
            out = tf.tanh(logits)
            return out

    # @define_scope
    def discriminator(self, x, n_units=128, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            h1 = tf.layers.dense(x, n_units, activation=None)
            h1 = tf.maximum(self.alpha*h1, h1)
            logits = tf.layers.dense(h1, 1, activation=None)
            out = tf.sigmoid(logits)
            
            return out, logits

    @define_scope
    def losses(self):
        g_model = self.generator(self.input_z, self.input_size, n_units=self.g_hidden_size)
        d_model_real, d_logits_real = self.discriminator(self.input_real, n_units=self.d_hidden_size)
        d_model_fake, d_logits_fake = self.discriminator(g_model, n_units=self.d_hidden_size, reuse=True)

        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_logits_real) * (1 - self.smooth))) 
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_logits_real)))
        d_loss = d_loss_real + d_loss_fake
        
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_logits_fake)))
        return  g_loss, d_loss

    @define_scope
    def d_opt(self):
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if var.name.startswith('losses/discriminator')]
        d_opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.losses[1], var_list=d_vars)
        return d_opt

    @define_scope
    def g_opt(self):
        t_vars = tf.trainable_variables()
        g_vars = [var for var in t_vars if var.name.startswith('losses/generator')]
        g_opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.losses[0], var_list=g_vars)
        return g_opt

def main ():
    input_size = 784 
    z_size = 100
    batch_size = 100
    epochs = 10

    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    input_real = tf.placeholder(tf.float32, [None, input_size], name='input_real')
    input_z = tf.placeholder(tf.float32, [None, z_size], name='input_z')

    model = Model(input_real, input_z, input_size, z_size, g_hidden_size=128, d_hidden_size=128, learning_rate=0.002, alpha=0.01, smooth=0.1)
    # g_model = model.generator
    # d_model = model.discriminator

    g_loss, d_loss = model.losses

    d_train_opt = model.d_opt
    g_train_opt = model.g_opt

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            for i in range(mnist.train.num_examples//batch_size):
                batch = mnist.train.next_batch(batch_size)
                batch_images = batch[0].reshape([batch_size, 784])
                batch_images = batch_images*2 - 1
                batch_z = np.random.uniform(-1, 1, size=[batch_size, z_size])

                sess.run(d_train_opt, feed_dict={input_real: batch_images, input_z: batch_z})
                sess.run(g_train_opt, feed_dict={input_z: batch_z})

            train_loss_d = d_loss.eval({input_z: batch_z, input_real: batch_images})
            train_loss_g = g_loss.eval({input_z: batch_z})
            
            print("Epoch {}/{}. Discriminator Loss: {}. Generator Loss: {}.".format(e+1, epochs, train_loss_d, train_loss_g))

if __name__ == '__main__':
    main()