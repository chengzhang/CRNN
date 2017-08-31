#coding:utf-8 

import logging
import numpy as np

## logging config
logging.basicConfig(
  level = logging.INFO,
  format = '[%(levelname)-8s %(asctime)-11s L%(lineno)-4d] %(message)s',
  datefmt = '%m-%d %H:%M')
logger = logging.getLogger()

def dense2sparse(dense): 
  indices, values = zip(*[([i, j], val) 
      for i, row in enumerate(dense) for j, val in enumerate(row)])
  max_len = max([len(row) for row in dense])
  shape = [len(dense), max_len] 
  sparse = (np.array(indices), np.array(values), np.array(shape)) 
  return sparse 

def sparse2dense(sparse): 
  (indices, values, shape) = sparse
  nparr = np.ndarray(shape, dtype=int) 
  default = max(values) + 1 if len(values) > 0 else 0
  nparr[:] = default 
  for i, v in zip(indices, values):
    nparr[tuple(i)] = v 
  dense = [[v for v in row if v != default] for row in nparr] 
  return dense 

