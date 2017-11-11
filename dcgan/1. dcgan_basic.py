import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data')

def model_inputs(real_dim, z_dim):
    inputs_real = tf.placeholder(tf.float32, [None, real_dim], name='input_real')
    inputs_z = tf.placeholder(tf.float32, [None, z_dim], name='input_z')

    return inputs_real, inputs_z

# Generator Network
def generator(z, out_dim, reuse=False, alpha=0.01, training=True):
    with  tf.variable_scope('generator', reuse=reuse):
        fc_1 = tf.layers.dense(z, 1024)
        fc_1 = tf.layers.batch_normalization(fc_1, training=training)
        fc_1 = tf.maximum(alpha*fc_1, fc_1)

        fc_2 = tf.layers.dense(fc_1, 7*7*64)
        fc_2 = tf.layers.batch_normalization(fc_2, training=training)
        fc_2 = tf.maximum(alpha*fc_2, fc_2)
        fc_2 = tf.reshape(fc_2, [-1, 7, 7, 64])
        # 7*7*64

        conv_1 = tf.layers.conv2d_transpose(fc_2, 32, 5, strides=2, padding='same')
        conv_1 = tf.layers.batch_normalization(conv_1, training=training)
        conv_1 = tf.maximum(alpha*conv_1, conv_1)
        # 14*14*32

        conv_2 = tf.layers.conv2d_transpose(conv_1, 1, 5, strides=2, padding='same')
        logits = tf.maximum(alpha*conv_2, conv_2)
        # 28*28*1

        out = tf.tanh(logits)
        return out

# Discriminator Network
def discriminator(x, reuse=False, alpha=0.01):
    with tf.variable_scope('discriminator', reuse=reuse):
        x = tf.reshape(x, [-1, 28, 28, 1])
        conv_1 = tf.layers.conv2d(x, 64, 5, strides=2, padding='same')
        conv_1 = tf.maximum(alpha*conv_1, conv_1)
        #14*14*64

        conv_2 = tf.layers.conv2d(x, 128, 5, strides=2, padding='same')
        conv_2 = tf.layers.batch_normalization(conv_2, training=True)
        conv_2 = tf.maximum(alpha*conv_2, conv_2)
        #7*7*128

        conv_2 = tf.reshape(conv_2, [-1, 7*7*128])
        logits = tf.layers.dense(conv_2, 1)
        out = tf.sigmoid(logits)
        
        return out, logits

#Plotting Samples
def view_samples(epoch, samples):
    fig, axes = plt.subplots(figsize=(7,7), nrows=4, ncols=4, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples):
        ax.axis('off')
        im = ax.imshow(img.reshape((28,28)), cmap='Greys_r')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig('dcgan_mnist_epoch_' + str(epoch+1) + '.png')
    plt.close('all')
    return fig, axes

# Hyperparameters
input_size = 784
z_size = 100
out_dim = 1
learning_rate = 0.001
alpha = 0.2
beta1=0.5
smooth = 0.1

# Losses
input_real, input_z = model_inputs(input_size, z_size)
g_model = generator(input_z, out_dim, alpha=alpha)
d_model_real, d_logits_real = discriminator(input_real, alpha=alpha)
d_model_fake, d_logits_fake = discriminator(g_model, reuse=True, alpha=alpha)

d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_logits_real) * (1 - smooth)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_logits_real)))
d_loss = d_loss_real + d_loss_fake
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_logits_fake)))

#Optimizers
t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
g_vars = [var for var in t_vars if var.name.startswith('generator')]

extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
    g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

# Training
batch_size = 128
epochs = 50
losses = []
save_gen_sample_every = 1

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        for i in range(mnist.train.num_examples//batch_size):
            batch = mnist.train.next_batch(batch_size)
            batch_images = batch[0].reshape([batch_size, 784])
            batch_images = batch_images*2 - 1
            batch_z = np.random.uniform(-1, 1, size=[batch_size, z_size])

            sess.run(d_train_opt, feed_dict={input_real: batch_images, input_z: batch_z})
            sess.run(g_train_opt, feed_dict={input_z: batch_z, input_real: batch_images})
        
        train_loss_d = d_loss.eval({input_z: batch_z, input_real: batch_images})
        train_loss_g = g_loss.eval({input_z: batch_z})

        print("Epoch {}/{}. Discriminator Loss: {}. Generator Loss: {}.".format(e+1, epochs, train_loss_d, train_loss_g))

        losses.append((train_loss_d, train_loss_g))
        sample_z = np.random.uniform(-1, 1, size=[16, z_size])
        if ((e)%save_gen_sample_every == 0):
            gen_samples = sess.run(generator(input_z, out_dim, reuse = True, alpha=alpha), feed_dict={input_z: sample_z})
            view_samples(e, gen_samples)

# Plotting Losses
fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator')
plt.plot(losses.T[1], label='Generator')
plt.title("Training Losses")
plt.legend()
plt.savefig('training_losses.png')