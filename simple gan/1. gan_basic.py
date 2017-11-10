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
def generator(z, out_dim, n_units=128, reuse=False, alpha=0.01):
    with  tf.variable_scope('generator', reuse=reuse):
        h1 = tf.layers.dense(z, n_units, activation=None)
        h1 = tf.maximum(alpha*h1, h1)
        logits = tf.layers.dense(h1, out_dim, activation=None)
        out = tf.tanh(logits)
        return out

# Discriminator Network
def discriminator(x, n_units=128, reuse=False, alpha=0.01):
    with tf.variable_scope('discriminator', reuse=reuse):
        h1 = tf.layers.dense(x, n_units, activation=None)
        h1 = tf.maximum(alpha*h1, h1)
        logits = tf.layers.dense(h1, 1, activation=None)
        out = tf.sigmoid(logits)
        return out, logits

#Plotting Samples
def view_samples(epoch, samples):
    fig, axes = plt.subplots(figsize=(7,7), nrows=4, ncols=4, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((28,28)), cmap='Greys_r')
    plt.savefig('epoch-' + str(epoch+1) + '.png')
    plt.close('all')
    return fig, axes

# Hyperparameters
input_size = 784
z_size = 100
g_hidden_size = 128
d_hidden_size = 128
alpha = 0.01
smooth = 0.1

# Losses
input_real, input_z = model_inputs(input_size, z_size)
g_model = generator(input_z, input_size, n_units=g_hidden_size, alpha=alpha)
d_model_real, d_logits_real = discriminator(input_real, n_units=d_hidden_size, alpha=alpha)
d_model_fake, d_logits_fake = discriminator(g_model, n_units=d_hidden_size, reuse=True, alpha=alpha)

d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_logits_real) * (1 - smooth)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_logits_real)))
d_loss = d_loss_real + d_loss_fake
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_logits_fake)))

#Optimizers
learning_rate = 0.001

t_vars = tf.trainable_variables()
g_vars = [var for var in t_vars if var.name.startswith('generator')]
d_vars = [var for var in t_vars if var.name.startswith('discriminator')]

g_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)
d_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=d_vars)

# Training
batch_size = 100
epochs = 50
losses = []
save_gen_sample_every = 10

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

        losses.append((train_loss_d, train_loss_g))
        sample_z = np.random.uniform(-1, 1, size=[16, z_size])
        gen_samples = sess.run(generator(input_z, input_size, n_units=g_hidden_size, reuse = True, alpha=alpha), feed_dict={input_z: sample_z})
        if ((e)%save_gen_sample_every == 0):
            view_samples(e, gen_samples)

# Plotting Losses
fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator')
plt.plot(losses.T[1], label='Generator')
plt.title("Training Losses")
plt.legend()
plt.savefig('training_losses.png')

