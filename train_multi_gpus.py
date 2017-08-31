#coding:utf-8 

import os
import sys
import logging
import argparse
import tensorflow as tf

from dataset import Dataset 
from multi_gpus import MultiGpusCRNN
import utils

## logging config
logging.basicConfig(
  level = logging.INFO,
  format = '[%(levelname)-8s %(asctime)-11s L%(lineno)-4d] %(message)s',
  datefmt = '%m-%d %H:%M')

def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    logging.info('Storing checkpoint {} to {}.'.format(step, logdir))
    sys.stdout.flush()
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)

def main(args):
  ## build model 
  optimizer = tf.train.AdamOptimizer(1e-4)
  mg_crnn = MultiGpusCRNN(args.n_gpus, optimizer, 
      batch_size=args.batch_size/args.n_gpus, 
      with_spatial_transform=args.with_spatial_transform)
  objective = mg_crnn.objective
  loss = mg_crnn.loss 
  error = mg_crnn.error
  predict = mg_crnn.predict
  probability = mg_crnn.log_prob 

  ## build dataset 
  dataset = Dataset(args.img_path_prefix, args.img_list_file, args.label_list_file,
      args.lexicon_file, val_ratio=args.val_ratio, test_ratio=args.test_ratio) 
  n_train_batches = dataset.n_train_samples / args.batch_size 
  n_val_batches = dataset.n_val_samples / args.batch_size 
  n_test_batches = dataset.n_test_samples / args.batch_size 

  ## start session, to fit the model 
  config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
  with tf.Session(config=config) as sess:
    ## init parameters 
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(var_list=tf.trainable_variables(), 
        max_to_keep=args.max_checkpoints)

    for i in xrange(n_train_batches * args.n_epochs):
      # {m0.image, m0.label, m1.image, m1.label, ...}
      batches = [dataset.next_train_batch(args.batch_size/args.n_gpus) 
          for j in range(args.n_gpus)] 
      feed_dict = mg_crnn.feed_dict(batches) 
      result = sess.run([objective, loss, error], feed_dict=feed_dict)

      if (i+1) % args.validate_every == 0:
        logging.info('step {}, epoch {}, loss {}, train err {}'.format( 
            i+1, batches[-1].epoch_id, result[1], result[2]))
        ## evaluate on dev set 
        if n_val_batches > 0: 
          val_batches = [dataset.next_val_batch(args.batch_size/args.n_gpus) 
              for j in range(args.n_gpus)] 
          feed_dict = mg_crnn.feed_dict(val_batches) 
          val_result = sess.run([predict, error], feed_dict=feed_dict)
          ## val_result[pred/err][gpu][0] 
          pred = [sample for j in range(args.n_gpus)
              for sample in utils.sparse2dense((val_result[0][j].indices, 
              val_result[0][j].values, val_result[0][j].dense_shape)) ]
          label = [sample for j in range(args.n_gpus)
              for sample in utils.sparse2dense(val_batches[j].labels) ]
          n_exact_matches = sum([1 for p, t in zip(pred, label) if p == t])  
          accuracy = n_exact_matches / float(args.batch_size) 
          logging.info('val err {}, acc {}'.format(val_result[1], accuracy))
          logging.info('val last sample pred:  {}'.format(pred[-1]))
          logging.info('val last sample label: {}'.format(label[-1]))

      if (i+1) % args.checkpoint_every == 0: 
        save(saver, sess, args.checkpoint_dir, i)

    ## evaluate on test set 
    if n_test_batches > 0: 
      test_err = 0
      for i in xrange(n_test_batches):
        batches = [dataset.next_test_batch(args.batch_size/arg.n_gpus) 
            for j in range(args.n_gpus)] 
        feed_dict = mg_crnn.feed_dict(batches) 
        batch_err = sess.run([error], feed_dict=feed_dict)
        test_err += batch_err[0]
      test_err /= n_test_batches
      logging.info('Test error {}'.format(test_err))

if __name__ == '__main__':
  def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path_prefix', default='./data/image', 
        help='Prefix of image path')
    parser.add_argument('--img_list_file', default='./data/image_list.txt', 
        help="Contain images' path")
    parser.add_argument('--label_list_file', default='./data/label_list.txt', 
        help="ontain images' label")
    parser.add_argument('--lexicon_file', default='./data/lexicon.txt', 
        help='Vocabulary, one word per line')
    parser.add_argument('--with_spatial_transform', type=bool, default=False) 
    parser.add_argument('--n_gpus', type=int, default=1) 
    parser.add_argument('--batch_size', type=int, default=128) 
    parser.add_argument('--n_epochs', type=int, default=20) 
    parser.add_argument('--val_ratio', type=float, default=0.2) 
    parser.add_argument('--test_ratio', type=float, default=0.1) 
    parser.add_argument('--validate_every', type=int, default=100) 
    parser.add_argument('--checkpoint_every', type=int, default=2000) 
    parser.add_argument('--checkpoint_dir', default='./checkpoint') 
    parser.add_argument('--max_checkpoints', type=int, default=10) 
    args = parser.parse_args()
    return args
  args = _get_args() 

  main(args)

