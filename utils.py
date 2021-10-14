"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

utils.py

(1) train_test_divide: Divide train and test data for both original and synthetic data.
(2) extract_time: Returns Maximum sequence length and each sequence length.
(3) rnn_cell: Basic RNN Cell.
(4) random_generator: random vector generator
(5) batch_generator: mini-batch generator
"""

## Necessary Packages
import numpy as np
from medgen.data_access.preprocessing import MissingnessDeltaT
from sklearn.model_selection import train_test_split
import tensorflow as tf


def train_test_divide (data_x, data_x_hat, data_t, data_t_hat, train_rate = 0.8):
  """Divide train and test data for both original and synthetic data.
  
  Args:
    - data_x: original data
    - data_x_hat: generated data
    - data_t: original time
    - data_t_hat: generated time
    - train_rate: ratio of training data from the original data
  """
  # Divide train/test index (original data)
  no = len(data_x)
  idx = np.random.permutation(no)
  train_idx = idx[:int(no*train_rate)]
  test_idx = idx[int(no*train_rate):]
    
  train_x = [data_x[i] for i in train_idx]
  test_x = [data_x[i] for i in test_idx]
  train_t = [data_t[i] for i in train_idx]
  test_t = [data_t[i] for i in test_idx]      
    
  # Divide train/test index (synthetic data)
  no = len(data_x_hat)
  idx = np.random.permutation(no)
  train_idx = idx[:int(no*train_rate)]
  test_idx = idx[int(no*train_rate):]
  
  train_x_hat = [data_x_hat[i] for i in train_idx]
  test_x_hat = [data_x_hat[i] for i in test_idx]
  train_t_hat = [data_t_hat[i] for i in train_idx]
  test_t_hat = [data_t_hat[i] for i in test_idx]
  
  return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat


def extract_time (data):
  """Returns Maximum sequence length and each sequence length.
  
  Args:
    - data: original data
    
  Returns:
    - time: extracted time information
    - max_seq_len: maximum sequence length
  """
  time = list()
  max_seq_len = 0
  for i in range(len(data)):
    max_seq_len = max(max_seq_len, len(data[i][:,0]))
    time.append(len(data[i][:,0]))
    
  return time, max_seq_len


def rnn_cell(module_name, hidden_dim):
  """Basic RNN Cell.
    
  Args:
    - module_name: gru, lstm, or lstmLN
    
  Returns:
    - rnn_cell: RNN Cell
  """
  assert module_name in ['gru','lstm','lstmLN']
  
  # GRU
  if (module_name == 'gru'):
    rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh)
  # LSTM
  elif (module_name == 'lstm'):
    rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, activation=tf.nn.tanh)
  # LSTM Layer Normalization
  elif (module_name == 'lstmLN'):
    rnn_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=hidden_dim, activation=tf.nn.tanh)
  return rnn_cell


def random_generator (batch_size, z_dim, labels_mb, T_mb, max_seq_len):
  """Random vector generation.
  
  Args:
    - batch_size: size of the random vector
    - z_dim: dimension of random vector
    - labels_mb: mini-batch of labels to concatenate to random vector
    - T_mb: time information for the random vector
    - max_seq_len: maximum sequence length
    
  Returns:
    - Z_mb: generated random vector
  """
  Z_mb = list()
  for i in range(batch_size):
    temp = np.zeros([max_seq_len, z_dim])
    temp_Z = np.random.uniform(0., 1, [T_mb[i], z_dim])
    temp[:T_mb[i],:] = temp_Z
    temp_Z_cat = np.concatenate((temp_Z, labels_mb[i]), 1)
    Z_mb.append(temp_Z_cat)
  return Z_mb


def batch_generator(data, labels, time, batch_size):
  """Mini-batch generator.
  
  Args:
    - data: time-series data
    - labels: labels for time series
    - time: time information
    - batch_size: the number of samples in each batch
    
  Returns:
    - X_mb: time-series data in each batch
    - labels_mb: labels in each batch
    - T_mb: time information in each batch
  """
  no = len(data)
  idx = np.random.permutation(no)
  train_idx = idx[:batch_size]     
            
  X_mb = list(np.concatenate((data[i], labels[i]), 1) for i in train_idx)
  labels_mb = list(labels[i] for i in train_idx)
  T_mb = list(time[i] for i in train_idx)
  
  return X_mb, labels_mb, T_mb

def reshape_synth_data(X_synth, y_synth):
  """
  Reshapes generated data for later evaluation.
  """
  random_state = np.random.RandomState(0)

  X_synth = np.transpose(X_synth, (0, 2, 1))

  X, m, delta_t = np.split(X_synth, 3, axis=1)
  # X = X_synth
  X, m, delta_t = MissingnessDeltaT().transform(X).values()
  X_stack = np.stack((X, m, delta_t), axis=1)

  val_fraction = 0.15
  test_fraction = 0.15
  train_fraction = 1.0 - val_fraction - test_fraction

  if len(y_synth.shape) == 1:  # stratify only if we have a single task
    y_strat = y_synth
  else:
    y_strat = None

  X_synth_train, X_intermed, y_synth_train, y_intermed = train_test_split(
    X_stack, y_synth,
    test_size=1 - train_fraction,
    random_state=random_state,
    stratify=y_strat)
  if y_strat is not None:
    y_intermed_strat = y_intermed
  else:
    y_intermed_strat = None
  X_synth_val, X_synth_test, y_synth_val, y_synth_test = train_test_split(
    X_intermed, y_intermed,
    test_size=test_fraction / (test_fraction + val_fraction),
    random_state=random_state,
    stratify=y_intermed_strat)
  X_synth_dict = {
    'X_train': X_synth_train[:, 0, ...],
    'X_val': X_synth_val[:, 0, ...],
    'X_test': X_synth_test[:, 0, ...],
    'm_train': X_synth_train[:, 1, ...],
    'm_val': X_synth_val[:, 1, ...],
    'm_test': X_synth_test[:, 1, ...],
    'delta_t_train': X_synth_train[:, 2, ...],
    'delta_t_val': X_synth_val[:, 2, ...],
    'delta_t_test': X_synth_test[:, 2, ...]
  }
  y_synth_dict = {
    'y_train': y_synth_train,
    'y_val': y_synth_val,
    'y_test': y_synth_test,
  }

  return X_synth_dict, y_synth_dict

def get_synth_labels(labels_orig, split=None):
  if split is None:
    return labels_orig
  else:
    ### Augment mode ###
    # curr_split = np.sum(labels_orig) / (labels_orig.shape[0] * labels_orig.shape[1])
    curr_split = 0.007
    tot_num_samples = len(labels_orig)
    p_have = int(curr_split * tot_num_samples)
    n_have = tot_num_samples - p_have
    p_add = int((split / (1 - split)) * n_have - p_have)
    labels = np.ones(p_add)
    ####################
    # labels = np.random.choice([0., 1.], size=len(labels_orig), p=[1-split, split])
    # Expand dims
    seq_len = labels_orig.shape[1]
    labels = np.transpose(np.tile(labels, (1, seq_len, 1)), (2,1,0)).astype(np.float32)
    return labels