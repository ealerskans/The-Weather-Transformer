"""
Train the model using the train and validation datasets.
"""

#====================
# Make deterministic
#====================
from mingpt.utils import set_seed
set_seed(42)

#==========================
# Standard library imports
#==========================
import warnings
# Silence FutureWarnings (something with my numpy version)
warnings.simplefilter(action='ignore', category=FutureWarning)
import datetime
import math
import numpy as np
import os
import pandas as pd
import pickle
import sys
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset

#===============
# Local imports
#===============
from Dataset import ForecastDataset
from general import InitializeFeatures, SpecifyFeatures, SpecifyRun, SpecifyModel, SpecifyDatasetFile
from global_parameters import *
from mingpt.model import GPT, GPTConfig
from mingpt.trainer import Trainer, TrainerConfig
from read_weather_data import read_training_datasets, read_val_datasets




if __name__ == '__main__':
    # Set up logging
    import logging
    logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
    )


    # Logicals for saving
    Save_Run = True
    Save_Model = True


    # Feature inclusion
    args = sys.argv
    include_features, feature_indices = InitializeFeatures(args)

    # Specify features
    features, Nfeatures = SpecifyFeatures(include_features)
    # Specify dataset file name
    dataset_file_generic, dataset_file_train, dataset_file_val, dataset_file_test = \
              SpecifyDatasetFile(data_dir,include_features,nstation_train,nstation_val,nstation_test,input_days)
    # Specify run
    run_dir = SpecifyRun(project_dir,Save_Run,features,nstation_train,nstation_val,n_epochs, \
                         batch_size,input_days,loss_metrics)
    # Specify model
    model_name = SpecifyModel(model_dir,include_features,features,nstation_train,nstation_val,n_epochs, \
                              batch_size,input_days,loss_metrics)



    print('Load Datasets')
    print(' ---Train data')
    read_dict={'data_train':False,'data_train_1d':True,'extra_data_train':False, \
               'data_raw_train':False,'extra_data_raw_train':False, 'station_train_index_shifted_1d':True}
    _, data_train_1d, _, _, _, station_train_index_shifted_1d = read_training_datasets(dataset_file_train,read_dict)

    print(' ---Validation data')
    read_dict={'data_val':False,'data_val_1d':True,'extra_data_val':False, \
               'data_raw_val':False,'extra_data_raw_val':False, 'station_val_index_shifted_1d':True}
    _, data_val_1d, _, _, _, station_val_index_shifted_1d = read_val_datasets(dataset_file_val,read_dict)


    print('Train and val forecast datasets')
    Nfeatures = len(feature_indices)
    t0 = time.time()
    train_dataset = ForecastDataset(data_train_1d, station_train_index_shifted_1d, block_size, feature_indices)
    val_dataset = ForecastDataset(data_val_1d, station_val_index_shifted_1d, block_size, feature_indices)
    print(' ---Elapsed time ForecastDataset (train+val): ', time.time()  - t0, ' s.')

    del data_train_1d, data_val_1d, station_train_index_shifted_1d, station_val_index_shifted_1d


    # Initialize the model
    mconf = GPTConfig(train_dataset.block_size, n_layer=1, n_head=2, n_embd=64, metrics=loss_metrics, n_input=Nfeatures)
    model = GPT(mconf)


    # Initialize a trainer instance and kick off training
    if Save_Model:
        tconf = TrainerConfig(max_epochs=n_epochs, batch_size=batch_size, learning_rate=1e-3,
                              lr_decay=True, warmup_tokens=batch_size*20, 
                              final_tokens=2*len(train_dataset[0])*block_size,
                              num_workers=4, ckpt_path=model_name)
    else:
        tconf = TrainerConfig(max_epochs=n_epochs, batch_size=batch_size, learning_rate=1e-3,
                              lr_decay=True, warmup_tokens=batch_size*20, 
                              final_tokens=2*len(train_dataset[0])*block_size,
                              num_workers=4)
    trainer = Trainer(model, train_dataset, val_dataset, Save_Run, run_dir, tconf)
    trainer.train()

    if Save_Model:
        # Need to save the model
        checkpoint = torch.load(model_name)
        model.load_state_dict(checkpoint)
        torch.save(model,model_name)

