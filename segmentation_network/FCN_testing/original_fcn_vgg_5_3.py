# Author: Wang Duo
# This script contains the model of adapted fully convolutional network

"""Builds the FCN network.

Implements the inference/loss/training pattern for model building.

1. inference() - Builds the model as far as is required for running the network
forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow.python.platform
import tensorflow as tf

# The medical image segmentation dataset has 5 classes, 
#0=background,1=lumen,2=wall,3=calcium deposit,4=thrombus
NUM_CLASSES = 5

# The images are 128x128 pixels.
IMAGE_SIZE = 128
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
#grey image
num_channels = 1
#convolutional kernel size
patch_size = 3
#first layer feature map depth
depth = 64

#learning rate decay parameters
decay_rate=0.95
decay_step=5000


def inference(images,batch_size,dropout_prob):
  """Build the FCN model up to where it may be used for inference.

  Args:
    images: Images placeholder, from inputs().
    batch_size: Size of each training batch.
    dropout_prob:probability of units keeped in dropout procedure.

  Returns:
    weighted_logits: Output tensor with the computed logits.
    Regularizers:square sum of weights used for regularization
  """

  with tf.name_scope('conv1'):
    conv1_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, num_channels, depth], stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),name='weights')
    conv1_biases = tf.Variable(tf.zeros([depth]),name='biases')
    conv1_2_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, depth, depth], stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),name='weights')
    conv1_2_biases = tf.Variable(tf.zeros([depth]),name='biases')

    conv1 = tf.nn.conv2d(images, conv1_weights, [1, 1, 1, 1], padding='SAME')
    hidden1 = tf.nn.relu(conv1 + conv1_biases)
    conv1 = tf.nn.conv2d(hidden1, conv1_2_weights, [1, 1, 1, 1], padding='SAME')
    hidden1 = tf.nn.relu(conv1 + conv1_2_biases)

    pool1 = tf.nn.max_pool(hidden1,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')

  with tf.name_scope('conv2'):
    conv2_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, depth, depth*2], stddev=1.0 / math.sqrt(float(IMAGE_SIZE))),name='weights')
    conv2_biases = tf.Variable(tf.constant(1.0, shape=[depth*2]),name='biases')
    conv2_2_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, depth*2, depth*2], stddev=1.0 / math.sqrt(float(IMAGE_SIZE))),name='weights')
    conv2_2_biases = tf.Variable(tf.constant(1.0, shape=[depth*2]),name='biases')
    conv2_3_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, depth*2, depth*2], stddev=1.0 / math.sqrt(float(IMAGE_SIZE))),name='weights')
    conv2_3_biases = tf.Variable(tf.constant(1.0, shape=[depth*2]),name='biases')
    conv2_link_weights = tf.Variable(tf.truncated_normal(
      [1, 1, depth*2, NUM_CLASSES], stddev=1.0 / math.sqrt(float(IMAGE_SIZE))),name='weights')
    conv2_link_biases = tf.Variable(tf.constant(1.0, shape=[NUM_CLASSES]),name='biases')

    conv2 = tf.nn.conv2d(pool1, conv2_weights, [1, 1,1, 1], padding='SAME')
    hidden2 = tf.nn.relu(conv2 + conv2_biases)
    conv2 = tf.nn.conv2d(hidden2, conv2_2_weights, [1, 1,1, 1], padding='SAME')
    hidden2 = tf.nn.relu(conv2 + conv2_2_biases)
    conv2 = tf.nn.conv2d(hidden2, conv2_3_weights, [1, 1,1, 1], padding='SAME')
    hidden2 = tf.nn.relu(conv2 + conv2_3_biases)

    pool2 = tf.nn.max_pool(hidden2,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    conv2_link=tf.nn.conv2d(pool2, conv2_link_weights, [1, 1,1, 1], padding='SAME')
    hidden2_link=tf.nn.relu(conv2_link + conv2_link_biases)

  with tf.name_scope('conv3'):
    conv3_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, depth*2, depth*4], stddev=1.0 / math.sqrt(float(IMAGE_SIZE))),name='weights')
    conv3_biases = tf.Variable(tf.constant(1.0, shape=[depth*4]),name='biases')
    conv3_2_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, depth*4, depth*4], stddev=1.0 / math.sqrt(float(IMAGE_SIZE))),name='weights')
    conv3_2_biases = tf.Variable(tf.constant(1.0, shape=[depth*4]),name='biases')
    conv3_3_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, depth*4, depth*4], stddev=1.0 / math.sqrt(float(IMAGE_SIZE))),name='weights')
    conv3_3_biases = tf.Variable(tf.constant(1.0, shape=[depth*4]),name='biases')
    conv3_link_weights = tf.Variable(tf.truncated_normal(
      [1, 1, depth*4, NUM_CLASSES], stddev=1.0 / math.sqrt(float(IMAGE_SIZE))),name='weights')
    conv3_link_biases = tf.Variable(tf.constant(1.0, shape=[NUM_CLASSES]),name='biases')

    conv3 = tf.nn.conv2d(pool2, conv3_weights, [1, 1, 1, 1], padding='SAME')
    hidden3 = tf.nn.relu(conv3 + conv3_biases)
    conv3 = tf.nn.conv2d(hidden3, conv3_2_weights, [1, 1, 1, 1], padding='SAME')
    hidden3 = tf.nn.relu(conv3 + conv3_2_biases)
    conv3 = tf.nn.conv2d(hidden3, conv3_2_weights, [1, 1, 1, 1], padding='SAME')
    hidden3 = tf.nn.relu(conv3 + conv3_2_biases)

    pool3 = tf.nn.max_pool(hidden3,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    conv3_link=tf.nn.conv2d(pool3, conv3_link_weights, [1, 1,1, 1], padding='SAME')
    hidden3_link=tf.nn.relu(conv3_link + conv3_link_biases)

  with tf.name_scope('conv4'):
    conv4_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, depth*4, depth*8], stddev=1.0 / math.sqrt(float(IMAGE_SIZE))),name='weights')
    conv4_biases = tf.Variable(tf.constant(1.0, shape=[depth*8]),name='biases')
    conv4_2_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, depth*8, depth*8], stddev=1.0 / math.sqrt(float(IMAGE_SIZE))),name='weights')
    conv4_2_biases = tf.Variable(tf.constant(1.0, shape=[depth*8]),name='biases')

    
    conv4 = tf.nn.conv2d(pool3, conv4_weights, [1, 1, 1, 1], padding='SAME')
    hidden4 = tf.nn.relu(conv4 + conv4_biases)
    conv4 = tf.nn.conv2d(hidden4, conv4_2_weights, [1, 1, 1, 1], padding='SAME')
    hidden4 = tf.nn.relu(conv4 + conv4_2_biases)
    pool4 = tf.nn.max_pool(hidden4,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')

  with tf.name_scope('conv5'):
    conv5_weights = tf.Variable(tf.truncated_normal(
      [8, 8, depth*8, depth*8], stddev=1.0 / math.sqrt(float(IMAGE_SIZE))),name='weights')
    conv5_biases = tf.Variable(tf.constant(1.0, shape=[depth*8]),name='biases')
    conv5_2_weights = tf.Variable(tf.truncated_normal(
      [1, 1, depth*8, depth*8], stddev=1.0 / math.sqrt(float(IMAGE_SIZE))),name='weights')
    conv5_2_biases = tf.Variable(tf.constant(1.0, shape=[depth*8]),name='biases')

    conv5 = tf.nn.conv2d(pool4, conv5_weights, [1, 1, 1, 1], padding='SAME')
    hidden5 = tf.nn.relu(conv5 + conv5_biases)
    hidden5 = tf.nn.dropout(hidden5,dropout_prob)
    conv5 = tf.nn.conv2d(hidden5, conv5_2_weights, [1, 1, 1, 1], padding='SAME')
    hidden5 = tf.nn.relu(conv5 + conv5_2_biases)
    hidden5 = tf.nn.dropout(hidden5,dropout_prob)

  with tf.name_scope('predict16'):
    p16_weights = tf.Variable(tf.truncated_normal(
      [1,1, depth*8, NUM_CLASSES], stddev=1.0 / math.sqrt(float(IMAGE_SIZE))),name='weights')
    p16_biases = tf.Variable(tf.constant(1.0, shape=[NUM_CLASSES]),name='biases')
    p16 = tf.nn.conv2d(hidden5, p16_weights, [1, 1, 1, 1], padding='SAME')
    p16_map = tf.nn.relu(p16 + p16_biases)

  with tf.name_scope('deconv1'):
    deconv1_weights = tf.Variable(tf.truncated_normal(
      [4, 4, NUM_CLASSES, NUM_CLASSES], stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),name='weights')
    deconv1_biases = tf.Variable(tf.constant(1.0, shape=[NUM_CLASSES]),name='biases')

    deconv1 = tf.nn.conv2d_transpose(p16_map, deconv1_weights,[batch_size,int(IMAGE_SIZE/8),int(IMAGE_SIZE/8),NUM_CLASSES], [1, 2, 2, 1], padding='SAME')
    hidden5 = tf.nn.relu(deconv1 + deconv1_biases)
    hidden5_c = tf.add(hidden5,hidden3_link)

  with tf.name_scope('deconv2'):
    deconv2_weights = tf.Variable(tf.truncated_normal(
      [4, 4,  NUM_CLASSES, NUM_CLASSES], stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),name='weights')
    deconv2_biases = tf.Variable(tf.constant(1.0, shape=[NUM_CLASSES]),name='biases')

    deconv2 = tf.nn.conv2d_transpose(hidden5_c, deconv2_weights,[batch_size,int(IMAGE_SIZE/4),int(IMAGE_SIZE/4),NUM_CLASSES], [1, 2, 2, 1], padding='SAME')
    hidden6 = tf.nn.relu(deconv2 + deconv2_biases)
    hidden6_c = tf.add(hidden6,hidden2_link)	


  with tf.name_scope('deconv3'):
    deconv3_weights = tf.Variable(tf.truncated_normal(
      [8, 8, NUM_CLASSES, NUM_CLASSES], stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),name='weights')
    deconv3_biases = tf.Variable(tf.constant(1.0, shape=[NUM_CLASSES]),name='biases')
    deconv3 = tf.nn.conv2d_transpose(hidden6_c, deconv3_weights,[batch_size,IMAGE_SIZE,IMAGE_SIZE,NUM_CLASSES], [1, 4, 4, 1], padding='SAME')
    hidden7 = tf.nn.relu(deconv3 + deconv3_biases)
    shape = hidden7.get_shape().as_list()
    len_flat=shape[0]* shape[1] * shape[2]
    reshape = tf.reshape(hidden7, [len_flat, shape[3]])

  with tf.name_scope('softmax_linear'):
    weights = tf.Variable(
        tf.truncated_normal([shape[3], NUM_CLASSES],
                            stddev=1.0 ),
        name='weights')
    biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                         name='biases')
    logits = tf.matmul(reshape, weights) + biases
    class_weights=tf.constant([0.123,0.92,2.04,10.3,2.18])
    weighted_logits=tf.mul(logits,class_weights)

    regularizers = (tf.nn.l2_loss(conv1_weights) + tf.nn.l2_loss(conv1_biases) +
                  tf.nn.l2_loss(conv2_weights) + tf.nn.l2_loss(conv2_biases)+
			tf.nn.l2_loss(conv3_weights) + tf.nn.l2_loss(conv3_biases)+
			tf.nn.l2_loss(conv4_weights) + tf.nn.l2_loss(conv4_biases)+
			tf.nn.l2_loss(conv5_weights) + tf.nn.l2_loss(conv5_biases)+
			tf.nn.l2_loss(conv1_2_weights) + tf.nn.l2_loss(conv1_2_biases) +
                  tf.nn.l2_loss(conv2_2_weights) + tf.nn.l2_loss(conv2_2_biases)+
			tf.nn.l2_loss(conv3_2_weights) + tf.nn.l2_loss(conv3_2_biases)+
                  tf.nn.l2_loss(conv2_3_weights) + tf.nn.l2_loss(conv2_3_biases)+
			tf.nn.l2_loss(conv3_3_weights) + tf.nn.l2_loss(conv3_3_biases)+
			tf.nn.l2_loss(conv4_2_weights) + tf.nn.l2_loss(conv4_2_biases)+
			tf.nn.l2_loss(deconv1_weights) + tf.nn.l2_loss(deconv1_biases) +
                  tf.nn.l2_loss(deconv2_weights) + tf.nn.l2_loss(deconv2_biases)+
			tf.nn.l2_loss(deconv3_weights) + tf.nn.l2_loss(deconv3_biases))
		
  return weighted_logits,regularizers

def loss(logits, labels,batch_size_in,regularizers):
  """Calculates the loss from the logits and the labels.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].
    batch_size_in:batch_size of training
    regularizers: sq sum of weights from inference function

  Returns:
    loss: Loss tensor of type float.
  """
  # Convert from sparse integer labels in the range [0, NUM_CLASSES)
  # to 1-hot dense float vectors (that is we will have batch_size vectors,
  # each with NUM_CLASSES values, all of which are 0.0 except there will
  # be a 1.0 in the entry corresponding to the label).
  labels=tf.reshape(labels,[batch_size_in*IMAGE_SIZE*IMAGE_SIZE])
  batch_size = tf.size(labels)
  labels = tf.expand_dims(labels, 1)  
  max_label=tf.reduce_max(labels)
  indices = tf.expand_dims(tf.range(0, batch_size), 1)
  concated = tf.concat(1, [indices, labels])

  onehot_labels = tf.sparse_to_dense(
      concated, tf.pack([batch_size, NUM_CLASSES]), 1.0, 0.0)
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits,
                                                          onehot_labels,
                                                          name='xentropy')
  loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

  loss+=1e-5 * regularizers
  return loss


def training(loss, learning_rate):
  """Sets up the training Ops.

  Creates a summarizer to track the loss over time in TensorBoard.

  Creates an optimizer and applies the gradients to all trainable variables.

  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.

  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.

  Returns:
    train_op: The Op for training.
  """
  # Add a scalar summary for the snapshot loss.
  tf.scalar_summary(loss.op.name, loss)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Create the gradient descent optimizer with the given learning rate with decay.
  learning_rate_decay=tf.train.exponential_decay(learning_rate,global_step,decay_step,decay_rate,staircase=True)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate_decay)

  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op


def evaluation(logits, labels,batch_size_in):
  """Evaluate the quality of the logits at predicting the label.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).

  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label's is was in the top k (here k=1)
  # of all logits for that example.
  labels=tf.reshape(labels,[batch_size_in*IMAGE_SIZE*IMAGE_SIZE])
  correct = tf.nn.in_top_k(logits, labels, 1)
  
  # Return the number of true entries.
  return tf.reduce_sum(tf.cast(correct, tf.int32))

def predict(softmax_one_hot):
  values,indices=tf.nn.top_k(softmax_one_hot)
  return indices
