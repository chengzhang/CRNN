#coding:utf-8 

import os
import sys
import logging
import argparse
import tensorflow as tf

from model import CRNN 

## logging config
logging.basicConfig(
  level = logging.INFO,
  format = '[%(levelname)-8s %(asctime)-11s L%(lineno)-4d] %(message)s',
  datefmt = '%m-%d %H:%M')

class MultiGpusCRNN():
  def __init__(self, n_gpus, optimizer, **kwargs): 
    self._n_gpus = n_gpus 
    self._optimizer = optimizer

    self._models, self.objective, self.loss = self._multi_models(**kwargs)
    self.predict, self.log_prob = self._predict(self._models)
    self.error = self._error(self._models)

  def feed_dict(self, inputs):
    fd = {}
    for i in range(self._n_gpus):
      fd[self._models[i].image] = inputs[i].images
      fd[self._models[i].label] = inputs[i].labels
    return fd 

  def _predict(self, models):
    pred = [m.predict for m in models]
    prob = [m.log_prob for m in models]
    return pred, prob 

  def _error(self, models):
    error = tf.reduce_mean([m.error for m in models])
    return error

  def _multi_models(self, **kwargs):
    models = []
    ## Calculate the gradients for each model tower.
    tower_losses = []
    tower_grads = []
    with tf.variable_scope(tf.get_variable_scope()) as scope: 
      for i in xrange(self._n_gpus):
        with tf.device('/gpu:%d' % i):
          crnn = CRNN(**kwargs)
          tf.get_variable_scope().reuse_variables()
          ## Calculate the gradients for the batch of data on this CIFAR tower.
          grads = self._optimizer.compute_gradients(crnn.loss)
          models.append(crnn)
          tower_losses.append(crnn.loss) 
          ## Keep track of the gradients across all towers.
          tower_grads.append(grads)

    loss = tf.reduce_mean(tower_losses) 
    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grads = self._average_gradients(tower_grads)
    # Apply the gradients to adjust the shared variables.
    apply_grad = self._optimizer.apply_gradients(grads) 
    return models, apply_grad, loss  

  def _average_gradients(self, tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
      # Note that each grad_and_vars looks like the following:
      #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
      grads = []
      for g, _ in grad_and_vars:
        # Add 0 dimension to the gradients to represent the tower.
        expanded_g = tf.expand_dims(g, 0)
  
        # Append on a 'tower' dimension which we will average over below.
        grads.append(expanded_g)
  
      # Average over the 'tower' dimension.
      grad = tf.concat(axis=0, values=grads)
      grad = tf.reduce_mean(grad, 0)
  
      # Keep in mind that the Variables are redundant because they are shared
      # across towers. So .. we will just return the first tower's pointer to
      # the Variable.
      v = grad_and_vars[0][1]
      grad_and_var = (grad, v)
      average_grads.append(grad_and_var)
    return average_grads

