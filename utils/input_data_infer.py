# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import matplotlib.pyplot as plt
import os
import re
import tensorflow.python.platform
import sys
#import tarfile
#from IPython.display import display, Image
from scipy import ndimage
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

image_size = 128  # Pixel width and height.


def load(data_path,max_num_images):
  dataset = np.ndarray(
    shape=(max_num_images, image_size, image_size), dtype=np.float32)
  labels = np.ndarray(shape=(max_num_images, image_size, image_size), dtype=np.int32)
  data_folders=os.listdir(data_path);
  label_index = 0
  image_index = 0
  for data_folder in data_folders:
  ###load negative first;
    image_folder=data_path+data_folder+'/patch'
    label_folder=data_path+data_folder+'/label'
    image_filenames=os.listdir(image_folder)
    label_filenames=os.listdir(label_folder)
    image_filenames = sorted(image_filenames, key=lambda x: (int(re.sub('\D','',x)),x))
    label_filenames = sorted(label_filenames, key=lambda x: (int(re.sub('\D','',x)),x))
    for image in image_filenames:
      image_file = os.path.join(image_folder, image)
      try:
        image_data = ndimage.imread(image_file).astype(float)
        image_mean=np.mean(image_data)
	image_std=np.std(image_data)
	image_data=(image_data-image_mean)/image_std              
        if image_data.shape != (image_size, image_size):
          raise Exception('Unexpected image shape: %s' % image_file)
        dataset[image_index, :, :] = image_data
        image_index += 1
      except IOError as e:
        print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    for image in label_filenames:
      label_file = os.path.join(label_folder, image)
      try:
        label_data = ndimage.imread(label_file)
        if label_data.shape != (image_size, image_size):
          raise Exception('Unexpected label file shape: %s' % label_file)
        labels[label_index, :, :] = label_data
        label_index += 1
      except IOError as e:
        print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
  num_images = image_index
  num_labels = label_index
  if num_labels != num_images:
    raise Exception('num of images %d is not squal to the number of label files',num_images,num_labels)
  dataset = dataset[0:num_images, :, :]
  labels = labels[0:num_labels,:,:]
  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  print('Labels:', labels.shape)
  return dataset, labels
  
def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels



def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot



class DataSet(object):

  def __init__(self, images, labels, fake_data=False, one_hot=False,
               dtype=tf.float32):
    """Construct a DataSet.

    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    dtype = tf.as_dtype(dtype).base_dtype
    if dtype not in (tf.uint8, tf.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape,
                                                 labels.shape))
      self._num_examples = images.shape[0]
      print(images.shape[0],images.shape)
      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      assert images.shape[3] == 1
      #images = images.reshape(images.shape[0],
      #                        images.shape[1] * images.shape[2])
      #if dtype == tf.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        #images = images.astype(np.float32)
        #images = np.multiply(images, 1.0 / 255.0)

    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * image_size*image_size
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples 
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]

def convert_to_label(images):
  print("converting to label")
  label=np.copy(images)
  label[label>1]=1
  label=label.reshape(label.shape[0],label.shape[1]*label.shape[2])
  return label

def read_data_sets(train_dir, fake_data=False, one_hot=False, dtype=tf.float32):
  class DataSets(object):
    pass
  data_sets = DataSets()

  images,labels=load(train_dir,30000)
  #test_dataset,test_labels=load(test_path,500)
  #valid_dataset,valid_labels=load(validation_path,500)

  np.random.seed(133)
  
  images,labels = randomize(images,labels)
  #test_dataset,test_labels = randomize(test_dataset,test_labels)
  #valid_dataset,valid_labels = randomize(valid_dataset,valid_labels)
  '''
  plt.figure()
  plt.imshow(images[100,:,:])
  plt.figure()
  plt.imshow(labels[100,:,:])
  plt.figure()
  plt.imshow(images[10000,:,:])
  plt.figure()
  plt.imshow(labels[10000,:,:])
  plt.figure()
  plt.imshow(images[25000,:,:])
  plt.figure()
  plt.imshow(labels[25000,:,:])
  '''
  VALIDATION_SIZE = 200
  TEST_SIZE=100
  
  images=images.reshape(images.shape + (1,))
  labels=labels.reshape(labels.shape[0],image_size*image_size)
  print("label reshaped:",labels.shape)
  
  data_sets.test = DataSet(images, labels, dtype=dtype)
 
  return data_sets

  
#data_sets=read_data_sets('data_in_patch/')

