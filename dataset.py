#coding:utf-8 

import os
import cv2
import random
import numpy as np
import argparse
import logging
import utils
from collections import namedtuple 

## TODO:
## 0. pre-process images only once, if it's the speed bottleneck 
## 1. use tf queue to preload

## logging config
logging.basicConfig(
  level = logging.INFO,
  format = '[%(levelname)-8s %(asctime)-11s L%(lineno)-4d] %(message)s',
  datefmt = '%m-%d %H:%M')

CHAR_SET = '0123456789abcdefghijklmnopqrstuvwxyz'

def label2word(label):
  return ''.join([CHAR_SET[l] for l in label]) 

def word2label(word):
  return [CHAR_SET.index(c) for c in word] 

def read_list_file(list_file):
  with open(list_file, 'r') as f:
    lines = [line.strip('\n') for line in f.readlines()]
    return lines
  return []

def read_image(img_path, img_width=100, img_height=32): 
  ## raw shape: [height, width, channel]
  raw = cv2.imread(img_path)
  ## gray shape: [height, width]
  gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
  resized = cv2.resize(gray, (img_width, img_height))
  reshaped = np.reshape(resized, (img_height, img_width, 1)) 
  normalized = (reshaped - 128.0) / 128.0 
  return normalized

Batch = namedtuple('batch', ['images', 'labels', 'paths', 'epoch_id', 'batch_id'])

class Subset(): 
  def __init__(self, samples, shuffle=False):
    self._samples = samples
    self.n_samples = len(self._samples)
    self._index = 0
    self._epoch = 0
    self._batch = 0 
    self._shuffle = shuffle 

  def next_batch(self, batch_size):
    if self.n_samples < 1:
      logging.warn('Tring to get batch from an empty dataset!')
      return [], 0, 0
    batch = []
    for i in range(batch_size): 
      batch.append(self._samples[self._index]) 
      self._index = (self._index + 1) % self.n_samples 
      self._batch += 1 
      if self._index == 0 and self._shuffle: 
        self._batch = 0 
        self._epoch += 1 
        if self._shuffle: 
          perm = range(self.n_samples)
          random.shuffle(perm)
          self._samples = [self._samples[i] for i in perm] 
    return batch, self._epoch, self._batch 

class Dataset():
  def __init__(self, img_path_prefix, img_list_file, label_list_file, lexicon_file, 
      val_ratio=0, test_ratio=0, shuffle_per_epoch=True):
    """ Dataset constructor. 
    Args:
      img_path_prefix: A file consisted of image files' path prefix.  
      img_list_file: A file consisted of image files' path.  
      label_list_file: A file consisted of image files' label.  
      lexicon_file: Vocab file. 
      val_ratio: Ratio of instances used to make validation set. 
      test_ratio: Ratio of instances used to make test set.
      shuffle_per_epoch: Shuffle the train set before every epoch if True. 
    """ 
    self._img_path_prefix = img_path_prefix
    self._img_list_file = img_list_file
    self._label_list_file = label_list_file
    self._lexicon_file = lexicon_file 
    self._shuffle_per_epoch = shuffle_per_epoch
    self._val_ratio = val_ratio
    self._test_ratio = test_ratio
    assert(self._val_ratio <= 1.0), 'val_ratio must in [0,1]'
    assert(self._test_ratio <= 1.0), 'test_ratio must in [0,1]'
    assert(self._val_ratio + self._test_ratio <= 1.0), 'val+test must in [0,1]'

    self._get_datasets()  

    self._lexicon = read_list_file(self._lexicon_file)
    self.max_word_len = max([len(word) for word in self._lexicon])
    logging.info('max word len: {}'.format(self.max_word_len))

  def next_train_batch(self, batch_size=37):
    entries, epoch_id, batch_id = self._train_set.next_batch(batch_size) 
    images, labels, paths = self._preprocess_batch(entries)
    return Batch(images, labels, paths, epoch_id, batch_id)
  
  def next_val_batch(self, batch_size=37):
    entries, epoch_id, batch_id = self._val_set.next_batch(batch_size) 
    images, labels, paths = self._preprocess_batch(entries)
    return Batch(images, labels, paths, epoch_id, batch_id)

  def next_test_batch(self, batch_size=37):
    entries, epoch_id, batch_id = self._test_set.next_batch(batch_size) 
    images, labels, paths = self._preprocess_batch(entries)
    return Batch(images, labels, paths, epoch_id, batch_id) 

  def _preprocess_batch(self, entries):
    ## TODO: use namedtuple to access e[0], e['path'], e['label']
    paths, labels = zip(*entries)
    ## preprocess of image
    images = [read_image(p) for p in paths]
    ## preprocess of label 
    labels = [word2label(self._lexicon[l]) for l in labels] 
    #indices, values = zip(*[([n, t], char) 
    #    for n, word in enumerate(labels) for t, char in enumerate(word)])
    #shape = [len(labels), max[len(l) for l in labels]] 
    #labels = (np.array(indices), np.array(values), np.array(shape)) 
    labels = utils.dense2sparse(labels) 
    return images, labels, paths  

  def _get_datasets(self): 
    """ Load dataset and split it into train, validation, and test set. """ 
    paths = read_list_file(self._img_list_file) 
    paths = [os.path.join(self._img_path_prefix, p) for p in paths]
    labels = read_list_file(self._label_list_file) 
    labels = [int(label) for label in labels] 
    assert(len(paths) != 0), '#img is 0!' 
    assert(len(paths) == len(labels)), '#img({}) != #label({})'.format(
        len(paths), len(labels))

    samples = zip(paths, labels) 
    self.n_samples = len(samples) 
    self.n_val_samples = int(self.n_samples * self._val_ratio) 
    self.n_test_samples = int(self.n_samples * self._test_ratio)
    self.n_train_samples = self.n_samples - self.n_val_samples - self.n_test_samples
    logging.info('#samples = {}, #train = {}, #val = {}, #test = {}'.format(
        self.n_samples, self.n_train_samples, self.n_val_samples, 
        self.n_test_samples))

    self._train_set = Subset(samples[:self.n_train_samples], self._shuffle_per_epoch) 
    self._val_set = Subset(
        samples[self.n_train_samples: self.n_train_samples + self.n_val_samples]) 
    self._test_set = Subset(samples[self.n_train_samples + self.n_val_samples:]) 

if __name__ == '__main__':
  """ A simple test on Dataset class. """ 

  def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path_prefix', default='./data/image', 
        help='Prefix of image path')
    parser.add_argument('--img_list_file', default='./data/image_list.txt', 
        help="Contain images' path")
    parser.add_argument('--label_list_file', default='./data/label_list.txt', 
        help="Contain images' label")
    parser.add_argument('--lexicon_file', default='./data/lexicon.txt', 
        help='Vocabulary, one word per line')
    args = parser.parse_args()
    return args
  args = _get_args() 

  dataset = Dataset(args.img_path_prefix, args.img_list_file, args.label_list_file,
      args.lexicon_file, val_ratio=0.2, test_ratio=0.1) 
  for i in range(10000):
    batch_size = random.randint(1, 100)
    loggin.info('try to get a batch with {} sampels'.format(batch_size))
    images, labels, paths, epoch_id, batch_id = dataset.next_train_batch(batch_size)
    images, labels, paths, epoch_id, batch_id = dataset.next_val_batch(batch_size)
    images, labels, paths, epoch_id, batch_id = dataset.next_test_batch(batch_size)

