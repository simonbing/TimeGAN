"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

main_timegan.py

(1) Import data
(2) Generate synthetic data
(3) Evaluate the performances in three ways
  - Visualization (t-SNE, PCA)
  - Discriminative score
  - Predictive score
"""

## Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import tensorflow as tf
from utils import reshape_synth_data
import os
import wandb
import warnings
warnings.filterwarnings("ignore")

# 1. TimeGAN model
from timegan import timegan
# 2. Data loading
from data_loading import real_data_loading, sine_data_generation, mimic_data_loading
# 3. Metrics
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics
from metrics.visualization_metrics import visualization


def main (args):
  """Main function for timeGAN experiments.
  
  Args:
    - data_name: sine, stock, or energy
    - seq_len: sequence length
    - Network parameters (should be optimized for different datasets)
      - module: gru, lstm, or lstmLN
      - hidden_dim: hidden dimensions
      - num_layer: number of layers
      - iteration: number of training iterations
      - batch_size: the number of samples in each batch
    - metric_iteration: number of iterations for metric computation
  
  Returns:
    - ori_data: original data
    - generated_data: generated synthetic data
    - metric_results: discriminative and predictive scores
  """
  ## Data loading
  if args.data_name in ['stock', 'energy']:
    ori_data = real_data_loading(args.data_name, args.seq_len)
  elif args.data_name == 'mimic':
    ori_data_train, ori_labels_train, ori_data_full, ori_labels_full = mimic_data_loading(args.features_path, args.labels_path)
  elif args.data_name == 'sine':
    # Set number of samples and its dimensions
    no, dim = 10000, 5
    ori_data, ori_labels = sine_data_generation(no, args.seq_len, dim)
    
  print(args.data_name + ' dataset is ready.')
    
  ## Synthetic data generation by TimeGAN
  # Set newtork parameters
  parameters = dict()  
  parameters['module'] = args.module
  parameters['hidden_dim'] = args.hidden_dim
  parameters['num_layer'] = args.num_layer
  parameters['iterations'] = args.iteration
  parameters['epochs'] = args.epochs
  parameters['batch_size'] = args.batch_size
  parameters['seed'] = args.seed
      
  generated_data, generated_labels = timegan(ori_data_train, ori_labels_train, ori_data_full, ori_labels_full, parameters)
  print('Finish Synthetic Data Generation')

  if args.save:
      # Save generated data
      if not os.path.isdir(args.out_path):
          os.makedirs(args.out_path)
      np.save(os.path.join(args.out_path, 'gen_feats.npy'), generated_data)
      np.save(os.path.join(args.out_path, 'gen_labels.npy'), generated_labels)
      print('Saved generated data!')
      # Reshape generated data
      X_synth_dict, y_synth_dict = reshape_synth_data(generated_data, generated_labels)
      np.savez(os.path.join(args.out_path,'X_synth.npz'), **X_synth_dict)
      np.savez(os.path.join(args.out_path,'y_synth.npz'), **y_synth_dict)
      print('Saved reshaped generated data!')
  
  ## Performance metrics   
  # Output initialization
  # metric_results = dict()
  #
  # # 1. Discriminative Score
  # discriminative_score = list()
  # for _ in range(args.metric_iteration):
  #   temp_disc = discriminative_score_metrics(ori_data, generated_data)
  #   discriminative_score.append(temp_disc)
  #
  # metric_results['discriminative'] = np.mean(discriminative_score)
  #
  # # 2. Predictive score
  # predictive_score = list()
  # for tt in range(args.metric_iteration):
  #   temp_pred = predictive_score_metrics(ori_data, generated_data)
  #   predictive_score.append(temp_pred)
  #
  # metric_results['predictive'] = np.mean(predictive_score)
  #
  # # 3. Visualization (PCA and tSNE)
  # visualization(ori_data, generated_data, 'pca')
  # visualization(ori_data, generated_data, 'tsne')
  #
  # ## Print discriminative and predictive scores
  # print(metric_results)

  metric_results = None

  return ori_data, generated_data, metric_results


if __name__ == '__main__':  
  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_name',
      choices=['mimic','sine','stock','energy'],
      default='stock',
      type=str)
  parser.add_argument(
      '--features_path',
      help='path to input features',
      default=None,
      type=str)
  parser.add_argument(
      '--labels_path',
      help='path to input labels',
      default=None,
      type=str)
  parser.add_argument(
      '--seq_len',
      help='sequence length',
      default=24,
      type=int)
  parser.add_argument(
      '--module',
      choices=['gru','lstm','lstmLN'],
      default='gru',
      type=str)
  parser.add_argument(
      '--hidden_dim',
      help='hidden state dimensions (should be optimized)',
      default=24,
      type=int)
  parser.add_argument(
      '--num_layer',
      help='number of layers (should be optimized)',
      default=3,
      type=int)
  parser.add_argument(
      '--iteration',
      help='Training iterations (should be optimized)',
      default=50000,
      type=int)
  parser.add_argument(
      '--epochs',
      help='Training epochs (should be optimized)',
      default=None,
      type=int)
  parser.add_argument(
      '--batch_size',
      help='the number of samples in mini-batch (should be optimized)',
      default=128,
      type=int)
  parser.add_argument(
      '--seed',
      help='random seed',
      default=0,
      type=int)
  parser.add_argument(
      '--out_path',
      help='path where to save generated data',
      default=None,
      type=str)
  parser.add_argument(
      '--run_name',
      help='wandb run nume',
      default=None,
      type=str)
  parser.add_argument(
      '--save',
      help='whether or not to save generated data',
      default=False,
      type=bool)
  parser.add_argument(
      '--metric_iteration',
      help='iterations of the metric computation',
      default=10,
      type=int)
  
  args = parser.parse_args()

  # init wandb logging
  config = dict(
      batch_size=args.batch_size,
      hidden_size=args.hidden_dim,
      pred_task="vent_bin",
      gen_model="timegan"
  )

  use_cuda = tf.test.is_gpu_available()
  if use_cuda:
      print('Running on GPU!')
  else:
      print('Running in CPU!')

  wandb.init(
      project='medgen',
      entity='bings',
      group='TimeGAN sweep',
      job_type='cluster' if use_cuda else 'local',
      mode='online' if use_cuda else 'offline',
      config=config
  )

  if args.run_name is not None:
      wandb.run.name = args.run_name
  
  # Calls main function  
  ori_data, generated_data, metrics = main(args)