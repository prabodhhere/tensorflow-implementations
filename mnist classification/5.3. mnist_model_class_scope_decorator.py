import functools
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

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
    def __init__(self, data, target, keep_prob):
        self.data = data
        self.target = target
        self.keep_prob = keep_prob
        self.logits
        self.optimizer
        self.accuracy

    @define_scope
    def logits(self):
        x_image = tf.reshape(self.data, [-1, 28, 28, 1], name='input')
        with tf.variable_scope('conv1'):
            W_conv1 = weight_variable([5, 5, 1, 32])
            b_conv1 = bias_variable([32])
            h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
            h_pool1 = max_pool_2x2(h_conv1)

        with tf.variable_scope('conv2'):
            W_conv2 = weight_variable([5, 5, 32, 64])
            b_conv2 = bias_variable([64])
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = max_pool_2x2(h_conv2)
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

        with tf.variable_scope('fc1'):
            W_fc1 = weight_variable([7*7*64, 1024])
            b_fc1 = bias_variable([1024])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        with tf.variable_scope('drop1'):
            h_fc1_drop1 = tf.nn.dropout(h_fc1, self.keep_prob)

        with tf.variable_scope('fc2'):
            W_fc2 = weight_variable([1024, 10])
            b_fc2 = bias_variable([10])
            h_fc2 = tf.matmul(h_fc1_drop1, W_fc2) + b_fc2
            return h_fc2

    @define_scope
    def optimizer(self):
        with tf.variable_scope('cross_entropy'):
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.target, logits=self.logits))
        with tf.variable_scope('train'):
            optimizer = tf.train.GradientDescentOptimizer(1e-4)
        return optimizer.minimize(cross_entropy)

    @define_scope
    def accuracy(self):
        with tf.variable_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.target, 1))
        with tf.variable_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy

def main():
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    data = tf.placeholder(tf.float32, [None, 784])
    target = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)

    model = Model(data, target, keep_prob)
    logits = model.logits
    optimizer = model.optimizer
    accuracy = model.accuracy

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            step = i+1
            batch_train_xs, batch_train_ys = mnist.train.next_batch(100)
            sess.run(optimizer, {data: batch_train_xs, target: batch_train_ys, keep_prob: 0.5})
            acc = sess.run(accuracy, {data: batch_train_xs, target: batch_train_ys, keep_prob: 0.5})
            if (step%100 == 0):
                print("Step: {}. Accuracy: {}".format(step, acc))
        print("Training done.")

        batch_size = 50
        num_batches = mnist.test.images.shape[0] // batch_size
        test_accuracy = 0
        for i in range(num_batches):
            batch_test_xs, batch_test_ys = mnist.test.next_batch(batch_size)
            test_accuracy += sess.run(accuracy, feed_dict={data: batch_test_xs, target: batch_test_ys, keep_prob: 1.0})
        test_accuracy /= num_batches

        print("Test Accuracy: {}".format(test_accuracy))

if __name__ == '__main__':
    main()