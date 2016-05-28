import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

image_size = 32  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

def load(data_path,max_num_images):
  dataset = np.ndarray(
    shape=(max_num_images, image_size, image_size), dtype=np.float32)
  labels = np.ndarray(shape=(max_num_images), dtype=np.int32)
  data_folders=os.listdir(data_path);
  label_index = 0
  image_index = 0
  for data_folder in data_folders:
  ###load negative first;
    folder=data_path+data_folder+'/neg'
    for image in os.listdir(folder):
      image_file = os.path.join(folder, image)
      try:
        image_data = (ndimage.imread(image_file).astype(float) -
                      pixel_depth / 2) / pixel_depth
        if image_data.shape != (image_size, image_size):
          raise Exception('Unexpected image shape: %s' % image_file)
        dataset[image_index, :, :] = image_data
        labels[image_index] = 0
        image_index += 1
      except IOError as e:
        print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    folder=data_path+data_folder+'/pos'
    for image in os.listdir(folder):
      image_file = os.path.join(folder, image)
      try:
        image_data = (ndimage.imread(image_file).astype(float) -
                      pixel_depth / 2) / pixel_depth
        if image_data.shape != (image_size, image_size):
          raise Exception('Unexpected image shape: %s' % image_file)
        dataset[image_index, :, :] = image_data
        labels[image_index] = 1
        image_index += 1
      except IOError as e:
        print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
  num_images = image_index
  dataset = dataset[0:num_images, :, :]
  labels = labels[0:num_images]
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

training_path='training_data/'
#test_path='test_data/'
#validation_path='validation_data/'

dataset,labels=load(training_path,4000)
#test_dataset,test_labels=load(test_path,500)
#valid_dataset,valid_labels=load(validation_path,500)

validation_set_size=100
test_set_size=100
np.random.seed(13)
dataset,labels = randomize(dataset,labels)
test_dataset=dataset[:test_set_size]
valid_dataset=dataset[test_set_size:test_set_size+validation_set_size]
train_dataset=dataset[test_set_size+validation_set_size:]

test_labels=labels[:test_set_size]
valid_labels=labels[test_set_size:test_set_size+validation_set_size]
train_labels=labels[test_set_size+validation_set_size:]


#train_dataset,train_labels = randomize(train_dataset,train_labels)
#test_dataset,test_labels = randomize(test_dataset,test_labels)
#valid_dataset,valid_labels = randomize(valid_dataset,valid_labels)

plt.figure()
plt.imshow(train_dataset[100,:,:])
print train_labels[100]
plt.figure()
plt.imshow(train_dataset[200,:,:])
print train_labels[200]
plt.figure()
plt.imshow(train_dataset[300,:,:])
print train_labels[300]


####save as pickle
pickle_file = 'ct_patch.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise
statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)
