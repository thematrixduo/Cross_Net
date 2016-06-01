# Author: Wang Duo
# This script use test networks trained for medical image segmentation.



# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
import tensorflow.python.platform
import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import scipy as sp
#load input and model scripts
import input_data_infer_5 as input_data
import 17l_freq_decay_5_1 as model
#import original_fcn_vgg_5_3 as model



# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 100, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 20, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', 'data_debug_5/', 'Directory to put the training data.')
flags.DEFINE_integer('num_class', 5, 'Number of classes.')
flags.DEFINE_integer('only_stats', True, 'If only to produce confusion matrix without saving predictions.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')

def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the input tensors.

  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.

  Args:
    batch_size: The batch size will be baked into both placeholders.

  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         model.IMAGE_SIZE,model.IMAGE_SIZE,1))
  labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size,
                                                         model.IMAGE_SIZE*model.IMAGE_SIZE))
  return images_placeholder, labels_placeholder


def fill_feed_dict(data_set, images_pl, labels_pl):
  """Fills the feed_dict for training the given step.

  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }

  Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    images_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().

  Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
  """

  # Create the feed_dict for the placeholders filled with the next
  # `batch size ` examples.
  images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size,
                                                 FLAGS.fake_data)
  #print("in fill_feed_dict labels indexes of >1:",numpy.where(labels_feed>0))
  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
  }
  return feed_dict

def predict(sess,
		logits,
		images_placeholder,
            	labels_placeholder,
            	data_set):
  """
  Run prediction for a number of CT image patches
  Args:
    sess: The session in which the model has been trained.
    logits: the model for inference
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().        
  """
  #calculate number of test batches and initialize confusion matrix
  steps_per_epoch=data_set.num_examples // FLAGS.batch_size
  num_img=0
  confusion_matrix=numpy.zeros((5,5))
  for step in xrange(steps_per_epoch):
    #get images and their true labels
    feed_dict = fill_feed_dict(data_set,
                               images_placeholder,
                               labels_placeholder)
    #feed image into the model and find most probable pixel label from one_hot probability vector
    softmax_one_hot,regularizers=sess.run(logits,feed_dict=feed_dict)
    batch_label=feed_dict[labels_placeholder]
    batch_image=feed_dict[images_placeholder]
    values,indices=sess.run(tf.nn.top_k(softmax_one_hot))
    t1=time.clock()
    #reshape predicted segmentation,image and ground truth into suitable format for image writing.
    for i in range(FLAGS.batch_size):
      num_img+=1
      predict=indices[i*model.IMAGE_SIZE*model.IMAGE_SIZE:(i+1)*model.IMAGE_SIZE*model.IMAGE_SIZE]
      predict=numpy.squeeze(predict)
      label=batch_label[i]
      label=numpy.squeeze(label)
      #update confusion matrix
      for row in range(FLAGS.num_class):
	for col in range(FLAGS.num_class):
	  count=numpy.size(label[numpy.where((predict==col)&(label==row))])
	  confusion_matrix[row,col]+=count

      label=label.reshape(model.IMAGE_SIZE,model.IMAGE_SIZE)   
      image=batch_image[i]
      image=numpy.squeeze(image)
      predict=predict.reshape(model.IMAGE_SIZE,model.IMAGE_SIZE)
      predicted_filename="prediction/predicted_label"+str(i)+".png"
      label_filename="prediction/label"+str(i)+".png"
      image_filename="prediction/image"+str(i)+".png"
      #if not only_stats, write image.
      if FLAGS.only_stats==False:
        sp.misc.imsave(predicted_filename,predict)
        sp.misc.imsave(image_filename,image)
        sp.misc.imsave(label_filename,label.astype(int))
  print("time used:",time.clock()-t1)
  print("number of test images:",num_img)
  return confusion_matrix
      


def run_infer():
  """Train model for a number of steps."""
  # Get the sets of images and labels for training, validation, and
  # test on model.
  data_sets = input_data.read_data_sets(FLAGS.train_dir, FLAGS.fake_data)

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Generate placeholders for the images and labels.
    images_placeholder, labels_placeholder = placeholder_inputs(
        FLAGS.batch_size)

    # Build a Graph that computes predictions from the inference model.
    logits = model.inference(images_placeholder,
			     FLAGS.batch_size,
				tf.constant(1.0)
				)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Run the Op to initialize the variables and load weights.
    init = tf.initialize_all_variables()
    sess.run(init)
    saver.restore(sess, "model_5_8l_1.ckpt")
    print("weights loaded")
    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.train.SummaryWriter('/tmp/model_logs_cnn',
                                            graph_def=sess.graph_def)

    # And then after everything is built, start inference.
    confusion_matrix=predict(sess,
            logits,
            images_placeholder,
            labels_placeholder,
            data_sets.test)
    print("Confusion Matrix:")
    print(confusion_matrix)
    class_accuracy=100*numpy.divide(numpy.diag(confusion_matrix),numpy.sum(confusion_matrix,axis=1))
    print("class accuracies",class_accuracy)
def main(_):
  run_infer()


if __name__ == '__main__':
  tf.app.run()
