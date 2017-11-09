import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
restore_checkpoint = False

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

def weight_variable(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')

def bias_variable(shape):
	return tf.Variable(tf.constant(0.1, shape=shape), name='bias')

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_pool')

def variable_summaries(var):
	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
		stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
		tf.summary.scalar('mean', mean)
		tf.summary.scalar('stddev', stddev)
		tf.summary.scalar('min', tf.reduce_min(var))
		tf.summary.scalar('max', tf.reduce_max(var))

x_image = tf.reshape(x, [-1, 28, 28, 1], name='input')

with tf.name_scope('conv1'):
	W_conv1 = weight_variable([5, 5, 1, 32])
	b_conv1 = bias_variable([32])
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	variable_summaries(W_conv1)
	variable_summaries(b_conv1)

with tf.name_scope('conv2'):
	W_conv2 = weight_variable([5, 5, 32, 64])
	b_conv2 = bias_variable([64])
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)
	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

with tf.name_scope('fc1'):
	W_fc1 = weight_variable([7*7*64, 1024])
	b_fc1 = bias_variable([1024])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

with tf.name_scope('drop1'):
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop1 = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope('fc2'):
	W_fc2 = weight_variable([1024, 10])
	b_fc2 = bias_variable([10])
	h_fc2 = tf.matmul(h_fc1_drop1, W_fc2) + b_fc2
	logits = h_fc2

global_step = tf.Variable(0, name='global_step', trainable=False)

with tf.name_scope('cross_entropy'):
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits))
with tf.name_scope('train'):
	train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)
with tf.name_scope('correct_prediction'):
	correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
with tf.name_scope('accuracy'):
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()
merged = tf.summary.merge_all()

with tf.Session() as sess:
	train_writer = tf.summary.FileWriter('./tmp/mnist/train', sess.graph)
	test_writer = tf.summary.FileWriter('./tmp/mnist/test')

	sess.run(tf.global_variables_initializer())
	if restore_checkpoint:
		saver.restore(sess, tf.train.latest_checkpoint('./models'))
	print("global step: {}".format(tf.train.global_step(sess, global_step)))
	for i in range(1000):
		step = i+1
		batch_train_xs, batch_train_ys = mnist.train.next_batch(50)
		if (step%100 == 0):
			summary, training_accuracy = sess.run([merged, accuracy], feed_dict={x: batch_train_xs, y_: batch_train_ys, keep_prob: 0.5})
			train_writer.add_summary(summary, step)
			print("Epoch {}, Train Accuracy: {}".format(i+1, training_accuracy))

		sess.run(train_step, feed_dict={x: batch_train_xs, y_: batch_train_ys, keep_prob: 0.5})

		if (step%500 == 0):
			saver.save(sess, './models/cnn_saver', global_step=global_step)

	print("Training done.")

	batch_size = 50
	num_batches = mnist.test.images.shape[0] // batch_size
	test_accuracy = 0
	for i in range(num_batches):
		batch_test_xs, batch_test_ys = mnist.test.next_batch(batch_size)
		test_accuracy += accuracy.eval(feed_dict={x: batch_test_xs, y_: batch_test_ys, keep_prob: 1.0})
	test_accuracy /= num_batches

	print("Test Accuracy: {}".format(test_accuracy))