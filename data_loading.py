"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

data_loading.py

(0) MinMaxScaler: Min Max normalizer
(1) sine_data_generation: Generate sine dataset
(2) real_data_loading: Load and preprocess real data
  - stock_data: https://finance.yahoo.com/quote/GOOG/history?p=GOOG
  - energy_data: http://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction
"""

## Necessary Packages
import numpy as np


def MinMaxScaler(data):
  """Min Max normalizer.
  
  Args:
    - data: original data
  
  Returns:
    - norm_data: normalized data
  """
  numerator = data - np.min(data, 0)
  denominator = np.max(data, 0) - np.min(data, 0)
  norm_data = numerator / (denominator + 1e-7)
  return norm_data


def sine_data_generation (no, seq_len, dim):
  """Sine data generation.
  
  Args:
    - no: the number of samples
    - seq_len: sequence length of the time-series
    - dim: feature dimensions
    
  Returns:
    - data: generated data
    - labels: randomly generated labels for the data
  """  
  # Initialize the output
  data = list()

  # Generate sine data
  for i in range(no):      
    # Initialize each time-series
    temp = list()
    # For each feature
    for k in range(dim):
      # Randomly drawn frequency and phase
      freq = np.random.uniform(0, 0.1)            
      phase = np.random.uniform(0, 0.1)
          
      # Generate sine signal based on the drawn frequency and phase
      temp_data = [np.sin(freq * j + phase) for j in range(seq_len)] 
      temp.append(temp_data)
        
    # Align row/column
    temp = np.transpose(np.asarray(temp))        
    # Normalize to [0,1]
    temp = (temp + 1)*0.5
    # Stack the generated data
    data.append(temp)

  # Sample labels
  labels = np.expand_dims(np.random.randint(2, size=no), 1)
                
  return data, labels
    

def real_data_loading (data_name, seq_len):
  """Load and preprocess real-world datasets.
  
  Args:
    - data_name: stock or energy
    - seq_len: sequence length
    
  Returns:
    - data: preprocessed data.
  """  
  assert data_name in ['stock','energy']
  
  if data_name == 'stock':
    ori_data = np.loadtxt('data/stock_data.csv', delimiter = ",",skiprows = 1)
  elif data_name == 'energy':
    ori_data = np.loadtxt('data/energy_data.csv', delimiter = ",",skiprows = 1)
        
  # Flip the data to make chronological data
  ori_data = ori_data[::-1]
  # Normalize the data
  ori_data = MinMaxScaler(ori_data)
    
  # Preprocess the dataset
  temp_data = []    
  # Cut data by sequence length
  for i in range(0, len(ori_data) - seq_len):
    _x = ori_data[i:i + seq_len]
    temp_data.append(_x)
        
  # Mix the datasets (to make it similar to i.i.d)
  idx = np.random.permutation(len(temp_data))    
  data = []
  for i in range(len(temp_data)):
    data.append(temp_data[idx[i]])
    
  return data

def mimic_data_loading(features_path, labels_path):
  """

  :param features_path:
  :param labels_path:
  :return:
  """
  X_dict = np.load(features_path)
  y_dict = np.load(labels_path)

  # Concatenate all input features
  X_list = list()
  m_list = list()
  delta_t_list = list()
  for file in X_dict.files:
    if file != 'feature_names' and not file.endswith('test'):
      if file.startswith('X'):
        X_list.append(X_dict[file])
      elif file.startswith('m'):
        m_list.append(X_dict[file])
      if file.startswith('delta_t'):
        delta_t_list.append(X_dict[file])
  X_concat_train = np.concatenate(X_list)
  m_concat_train = np.concatenate(m_list)
  delta_t_concat_train = np.concatenate(delta_t_list)

  data_concat_train = np.concatenate((X_concat_train, m_concat_train, delta_t_concat_train), axis=1)
  data_concat_train = np.transpose(data_concat_train, (0, 2, 1))

  # Full data (for generation)
  X_list.append(X_dict['X_test'])
  m_list.append(X_dict['m_test'])
  delta_t_list.append(X_dict['delta_t_test'])
  X_concat_full = np.concatenate(X_list)
  m_concat_full = np.concatenate(m_list)
  delta_t_concat_full = np.concatenate(delta_t_list)

  data_concat_full = np.concatenate((X_concat_full, m_concat_full, delta_t_concat_full), axis=1)
  data_concat_full = np.transpose(data_concat_full, (0, 2, 1))

  # Concatenate labels
  y_list = list()
  for file in y_dict.files:
    if not file.endswith('test'):
      y_list.append(y_dict[file])
  labels_concat_train = np.concatenate(y_list)

  y_list.append(y_dict['y_test'])
  labels_concat_full = np.concatenate(y_list)

  return data_concat_train, np.expand_dims(labels_concat_train, 1), data_concat_full, np.expand_dims(labels_concat_full, 1)