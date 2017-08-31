#coding:utf-8 

import logging
import numpy as np 
import tensorflow as tf
import utils
import spatial_transformer as st 

## logging config
logging.basicConfig(
  level = logging.INFO,
  format = '[%(levelname)-8s %(asctime)-11s L%(lineno)-4d] %(message)s',
  datefmt = '%m-%d %H:%M')

## notice: give each op with variable a name, to get the variable reused across
##         multi replica of the graph.

class CRNN():
  def __init__(self, batch_size, img_height=32, img_width=100, n_classes=37, 
      rnn_hidden_size=256, with_spatial_transform=False): 
    self._batch_size = batch_size 
    self._img_height = img_height
    self._img_width = img_width 
    self._n_classes = n_classes
    self._rnn_hidden_size = rnn_hidden_size 
    self._with_spatial_transform = with_spatial_transform 

    ## placeholder for crnn input 
    self.image = tf.placeholder('float', [batch_size, img_height, img_width, 1])
    self.label = tf.sparse_placeholder(tf.int32) 
    self._infer = self._inference(self.image) 
    self.loss = self._loss(self.label, self._infer) 
    self.predict, self.log_prob = self._predict(self._infer) 
    self.error = self._error(self.label, self.predict)  

  def _inference(self, x): 
    """ From image to logits. """
    ## x shape: [N, H, W, C=1]
    if self._with_spatial_transform:
      x = self._spatial_transform(x) 
    cnn = self._cnn(x)
    ## cnn shape: [N, H=1, W=T, C=512]
    cnn2rnn = self._cnn2rnn(cnn)
    ## cnn2rnn shape: [T, N, C] 
    rnn = self._rnn(cnn2rnn, self._rnn_hidden_size) 
    ## rnn shape: [T, N, O=2*rnn_hidden_size]
    proj = tf.contrib.layers.fully_connected(rnn, self._n_classes, scope='proj') 
    ## proj shape: [T, N, K=n_classes=n_labels+1] 
    return proj

  def _spatial_transform(self, x):  
    ## x shape: [N, W, H, C=1]
    conv1_loc = tf.layers.conv2d(x, 16, 3, padding='same', activation=tf.nn.relu,
        name='conv1_loc') 
    pool1_loc = tf.layers.max_pooling2d(conv1_loc, 2, 2)
    flat_loc = tf.contrib.layers.flatten(pool1_loc)
    fc1_loc = tf.contrib.layers.fully_connected(flat_loc, 64, scope='fc1_loc')
    ac1_loc = tf.nn.tanh(fc1_loc) 
    fc2_loc = tf.contrib.layers.fully_connected(ac1_loc, 6, scope='fc2_loc') 
    ac2_loc = tf.nn.tanh(fc2_loc) 
    stn = st.transformer(x, ac2_loc, out_size=(self._img_height, self._img_width)) 
    return stn 

  def _cnn(self, x):
    """ Convolutionnal Neural Network part """
    # x: [N, W, H, C]

    ## conv2d(inputs, filters, kernel_size)
    conv1 = tf.layers.conv2d(x, 64, 3, padding='same', activation=tf.nn.relu,
        name='conv1')
    ## max_pooling2d(inputs, pool_size, strides)
    pool1 = tf.layers.max_pooling2d(conv1, 2, 2)

    conv2 = tf.layers.conv2d(pool1, 128, 3, padding='same', activation=tf.nn.relu,
        name='conv2') 
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2)

    ## TODO: set activation=None? 
    conv3 = tf.layers.conv2d(pool2, 256, 3, padding='same', name='conv3')
    bn3 = tf.layers.batch_normalization(conv3, name='bn3')
    act3 = tf.nn.relu(bn3)

    conv4 = tf.layers.conv2d(act3, 256, 3, padding='same', activation=tf.nn.relu,
        name='conv4')
    pool4 = tf.layers.max_pooling2d(conv4, 2, (2,1), padding='same')

    conv5 = tf.layers.conv2d(pool4, 512, 3, padding='same', name='conv5')
    bn5 = tf.layers.batch_normalization(conv5, name='bn5')
    act5 = tf.nn.relu(bn5)

    conv6 = tf.layers.conv2d(act5, 512, 3, padding='same', activation=tf.nn.relu,
        name='conv6')
    pool6 = tf.layers.max_pooling2d(conv6, 2, (2,1), padding='same')

    conv7 = tf.layers.conv2d(pool6, 512, 2, name='conv7') 
    bn7 = tf.layers.batch_normalization(conv7, name='bn7')
    act7 = tf.nn.relu(bn7)

    return act7 

  def _cnn2rnn(self, x): 
    # x shape: [N, H=1, W=T, C]
    ## TODO: make no assumption on 'shape[1] == 1', 
    ##       combine shape[1] and shape[3], instead of squeezing shape[1] 
    x = tf.reshape(x, [-1, int(x.shape[2]), int(x.shape[3])])
    # x shape: [N, T, C]
    x = tf.transpose(x, perm=[1, 0, 2])
    # x shape: [T, N, C]
    return x

  def _rnn(self, x, hidden_size, n_layers=2):
    """ Bidirectionnal LSTM Recurrent Neural Network part """
    # x shape: [T, N, C]
    x = tf.unstack(x)
    # x shape: [N, C] * T
    for l in range(n_layers): 
      fw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0)
      bw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0)
      x, _, _ = tf.contrib.rnn.static_bidirectional_rnn(fw_cell, bw_cell, x, 
          dtype=tf.float32, scope=str(l))
    x = tf.stack(x)
    # x shape: [T, N, O=2*rnn_hidden_size] 
    return x 

  def _loss(self, labels, infer): 
    ## TODO: make no assumption on batch_size 
    input_len = np.ones(int(infer.shape[1])) * int(infer.shape[0]) 
    loss = tf.nn.ctc_loss(labels, infer, input_len)
    loss = tf.reduce_mean(loss)
    return loss 

  def _predict(self, infer): 
    input_len = np.ones(int(infer.shape[1])) * int(infer.shape[0]) 
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(infer, input_len)
    return decoded[0], log_prob 

  def _error(self, labels, pred): 
    error = tf.reduce_mean(tf.edit_distance(tf.cast(pred, tf.int32), labels))
    return error

if __name__ == '__main__': 
  """ A simple test to CRNN. """
  crnn = CRNN(batch_size=1, img_height=32, img_width=100, n_classes=37, 
      rnn_hidden_size=256) 
