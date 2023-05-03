"""
Predict t2m
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
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import sys
import time
import torch

#===============
# Local imports
#===============
from Dataset import TestDataset
from general import GetDataVariables, InitializeFeatures, SpecifyFeatures, \
                    SpecifyModel, SpecifyDatasetFile, SpecifyPredictionFile
from global_parameters import *
from mingpt.model import GPT, GPTConfig
from mingpt.trainer import Trainer, TrainerConfig
from mingpt.utils import predict, multi_predict
from read_weather_data import read_training_datasets, read_val_datasets, \
                              read_test_datasets, read_auxiliary_data




if __name__ == '__main__':
    # Start time taking:
    t0_tot = time.time()

    print('T2m predictions')


    # Read station IMEI, time period and lat,lon from file
    station_file = list_dir + 'station_list.csv'
    df = pd.read_csv(station_file)

    # Number of stations
    nstation = df.shape[0]
    station_idx = np.arange(nstation)

    # Get IMEI, start and end dates and lat,lon for all stations/locations
    imei = df['imei'].values
    start_dates = df['start_date'].values
    end_dates = df['end_date'].values
    latitudes = df['lat'].values
    longitudes = df['lon'].values



    # Logicals for saving
    Save_Predictions = True


    # Feature inclusion - either take command-line argument as input or use the ones specified here
    # See if there are any command-line arguments supplied
    args = sys.argv
    include_features, feature_indices = InitializeFeatures(args)


    # Specify features
    features, Nfeatures = SpecifyFeatures(include_features)
    # Specify dataset file name
    dataset_file_generic, dataset_file_train, dataset_file_val, dataset_file_test = \
             SpecifyDatasetFile(data_dir,include_features,nstation_train,nstation_val,nstation_test,input_days)
    # Specify the predictions file
    predictions_file = SpecifyPredictionFile(data_dir,include_features,nstation_train,nstation_val,nstation_test,n_epochs, \
                                             batch_size,input_days,loss_metrics)
    # Specify model
    model_name = SpecifyModel(model_dir,include_features,features,nstation_train,nstation_val,n_epochs, \
                              batch_size,input_days,loss_metrics)


    print('Load the model')
    model = torch.load(model_name,map_location=torch.device('cpu'))


    print('Load Datasets')
    print(' ---Generic data')
    mu, std, station_train, station_val, station_test = read_auxiliary_data(dataset_file_generic)

    print(' ---Test data')
    read_dict={'data_test':True,'data_test_1d':True,'extra_data_test':True, \
               'data_raw_test':True,'extra_data_raw_test':True}
    data_test, data_test_1d, extra_data_test, data_raw_test, extra_data_raw_test = read_test_datasets(dataset_file_test,read_dict)

    # Create test dataset
    t0 = time.time()
    test_dataset, analysis_dataset, extra_analysis_dataset\
              = TestDataset(data_test,data_raw_test,extra_data_raw_test,data_test_1d,feature_indices,nfc_input,fc_update, \
                            fc_output_interval,prediction_window,block_size,station_test,verbose=True)
    print(' ---Elapsed time ValidationDataset: ', time.time()  - t0, ' s.')

    # Delete already-used data
    del data_test, data_test_1d, extra_data_test, data_raw_test, extra_data_raw_test


    # Get the IMEI number of the test stations
    imei_test = [imei[istation] for istation in station_test]


    print('Make predictions')

    # Temperature lists
    t2m_pred = []
    t2m_nwp = []
    t2m_obs = []

    # Predictions for all test batches/forecasts at the same time
    for istation in range(nstation_test):
        print('Working on station no. ' + str(istation) + ' of ' + str(nstation_test) + '.')
        # Want to feed the following to the model:
        # - historical obs
        # - historical and future predictions from the model
        x_obs = torch.tensor(test_dataset[istation][:,:,0], dtype=torch.float).unsqueeze(-1)
        x_model = torch.tensor(test_dataset[istation][:,:,1:], dtype=torch.float)
        # Predictions
        y = multi_predict(model, x_obs, x_model, prediction_window)
        # ML predicted t2m (index 0 = pred t2m obs)
        t2m_pred_tmp = y[:,block_size:block_size+prediction_window,0].numpy() * std[0] + mu[0]  
        # Station/OBS t2m
        t2m_obs_tmp = analysis_dataset[istation][:,:,0].numpy() * std[0] + mu[0]
        # NWP t2m
        t2m_nwp_tmp = analysis_dataset[istation][:,:,1].numpy() * std[1] + mu[1]

        # Append data
        t2m_pred.append(t2m_pred_tmp)
        t2m_nwp.append(t2m_nwp_tmp)
        t2m_obs.append(t2m_obs_tmp)


    # Save the predictions
    if Save_Predictions:
        save_tuple = (t2m_nwp, t2m_obs, t2m_pred)

        outfile = open(predictions_file,'wb')
        pickle.dump(save_tuple,outfile)
        outfile.close()

    # Final time taking
    print('\n---Total elapsed time for predictions: ', time.time() - t0_tot, ' s.\n\n')
