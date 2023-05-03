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
import matplotlib.pyplot as plt
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
from Dataset import TestDataset
from general import GetDataVariables, SpecifyFeatures, \
                    SpecifyModel, SpecifyDatasetFile, SpecifyPredictionFile
from global_parameters import *
from mingpt.model import GPT, GPTConfig
from mingpt.trainer import Trainer, TrainerConfig
from mingpt.utils import predict
from read_weather_data import read_auxiliary_data, read_test_datasets




if __name__ == '__main__':
    # Logicals
    Plot_gradient_continuously = False

    # Plot settings
    colors = ['b','orange','g','r','m','k','y','c', \
              'b','orange','g','r','m','k','y','c', \
              'b','orange','g','r','m','k','y','c', \
              'b','orange','g','r','m','k','y','c']
    markers = ['x','o','^','v','*','+','p','d', \
               'o','^','v','*','+','p','d','x', \
               '^','v','*','+','p','d','x','o', \
               'v','*','+','p','d','x','o','^',]
    linestyles = ['--','--','--','--','--','--','--','--', \
                  'dotted','dotted','dotted','dotted','dotted','dotted','dotted','dotted', \
                  'dashdot','dashdot','dashdot','dashdot','dashdot','dashdot','dashdot','dashdot', \
                  '-.','-.','-.','-.','-.','-.','-.','-.']

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


    # Get dates in datetime format
    start_datetimes = [datetime.datetime.strptime(start_date,fmt_strp) for start_date in start_dates]
    end_datetimes = [datetime.datetime.strptime(end_date,fmt_strp) for end_date in end_dates]


    # Loicals for loading
    Load_data = False

    # Logicals for saving
    Save_data = True

    # Logicals for plotting
    Plot_dydx = False
    Plot_dydx_mean = False
    Plot_dydx_mean_all = True


    # Feature inclusion
    include_features = {}
    include_features['OBS_t2m'] = True
    include_features['NWP_t2m'] = True
    include_features['NWP_u10m'] = True
    include_features['NWP_v10m'] = True
    include_features['NWP_wspd10m'] = True
    include_features['NWP_sin_wsdir10m'] = True
    include_features['NWP_cos_wsdir10m'] = True
    include_features['NWP_rh2m'] = True
    include_features['NWP_q2m'] = True
    include_features['NWP_td2m'] = True
    include_features['NWP_mslp'] = True
    include_features['NWP_lhtfl'] = True
    include_features['NWP_shtfl'] = True
    include_features['NWP_nswrf'] = True
    include_features['NWP_nlwrf'] = True
    include_features['NWP_tcc'] = True
    include_features['NWP_tp'] = True
    include_features['NWP_pwat'] = True
    include_features['NWP_gh500'] = True
    include_features['NWP_gh700'] = True
    include_features['NWP_gh850'] = True
    include_features['NWP_t500'] = True
    include_features['NWP_t700'] = True
    include_features['NWP_t850'] = True
    include_features['NWP_u500'] = True
    include_features['NWP_u700'] = True
    include_features['NWP_u850'] = True
    include_features['NWP_v500'] = True
    include_features['NWP_v700'] = True
    include_features['NWP_v850'] = True
    include_features['NWP_w700'] = True
    include_features['TIME_sin_doy'] = False
    include_features['TIME_cos_doy'] = False
    include_features['TIME_sin_hod'] = False
    include_features['TIME_cos_hod'] = False
    include_features['NWP_wspd500'] = False
    include_features['NWP_wspd700'] = False
    include_features['NWP_wspd850'] = False
    include_features['NWP_sin_wsdir500'] = False
    include_features['NWP_cos_wsdir500'] = False
    include_features['NWP_sin_wsdir700'] = False
    include_features['NWP_cos_wsdir700'] = False
    include_features['NWP_sin_wsdir850'] = False
    include_features['NWP_cos_wsdir850'] = False

    # Feature indices
    feature_indices = [i for i,key in enumerate(include_features) if include_features[key] == True]

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



    if Load_data:
        print('Load already-calculated gradients')
        ofile = data_dir + 'gradients.pickle'
        outfile = open(ofile,'rb')
        save_tuple = pickle.load(outfile)
        dydx = save_tuple
        outfile.close()
        del save_tuple
    else:
        print('Load the model')
        model = torch.load(model_name,map_location=torch.device('cpu'))

        print('Load Datasets')
        print(' ---Generic data')
        mu, std, station_train, station_val, station_test = read_auxiliary_data(dataset_file_generic)

        print('Load test dataset')
        read_dict={'data_test':True,'data_test_1d':True,'extra_data_test':True, \
                   'data_raw_test':True,'extra_data_raw_test':True}
        data_test, data_test_1d, extra_data_test, data_raw_test, extra_data_raw_test = read_test_datasets(dataset_file_test,read_dict)
    
        # Create test dataset
        t0 = time.time()
        test_dataset, analysis_dataset, extra_analysis_dataset\
                  = TestDataset(data_test,data_raw_test,extra_data_raw_test,data_test_1d,feature_indices,nfc_input,fc_update, \
                                fc_output_interval,prediction_window,block_size,station_test,verbose=True)

        # Delete already-used data
        del data_test, data_test_1d, extra_data_test, data_raw_test, extra_data_raw_test

        print('Calculate gradients')
        dydx = []
        for istation in range(nstation_test):
            print('Working on station no ' + str(istation) + ' of ' + str(nstation_test))
            # Number of data points to make predictions for
            npredict = test_dataset[istation].shape[0]
    
            dydx_forecast = np.zeros((prediction_window,block_size,Nfeatures))
    
            for ipredict in range(npredict):
    #            print('--- Prediction ' + str(ipredict+1) + ' out of ' + str(npredict))
                dydx_tmp = np.zeros((prediction_window,block_size,Nfeatures))
    
                x_obs = torch.tensor(test_dataset[istation][ipredict,:,0], dtype=torch.float).unsqueeze(-1)
                x_model = torch.tensor(test_dataset[istation][ipredict,:,1:], dtype=torch.float)
    
                x = torch.cat((x_obs[:block_size,:],x_model[:block_size,:]), dim=1)
                x = x.unsqueeze(0)
                x_model = x_model.unsqueeze(0)
    
                # Loop over prediction window - Number of forecast outputs (+3h - +54h)
                for k in range(prediction_window):
                    x_cond = x.clone().detach() if x.size(1) <= block_size else x[:, -block_size:].clone().detach() # crop the sequence if needed
                    x_cond.requires_grad = True  # set pytorch to compute gradients for input
                    outputs, _ = model.forward(x_cond)
                    max_prediction = outputs.mean()  # must be scalar to compute gradient - mean seems to be the best
                                                     # max: Gives zeros for a lot of the input days and a sharp spike otherwise,
                                                     #      Starts with most important for recent days and then descends
                                                     # min: Gives zeros for a lot of the input days and a sharp spike otherwise
                                                     #      Starts with most important for oldest days and then descends - until it starts from most recent days
                                                     # median: Gives zeros for a lot of the input days and a sharp spike otherwise
                                                     # mean: Gives continuous importance/gradients
                    if k == 0:
                        max_prediction.backward(retain_graph=True)
                    else:
                        max_prediction.backward()
    
                    # Take the absolute of the gradient since I want to take the average later and gradients
                    # might cancel if they're opposite signs
                    dydx_tmp[k,:,:] = np.abs(x_cond.grad[0,:,:])  # same shape as input, telling you the "importance" of each dimension
    
                    if Plot_gradient_continuously:
                        fig, ax = plt.subplots(figsize=[10,8])
                        for i in range(Nfeatures):
                            ax.plot(dydx_tmp[k,:,i], color=colors[i], marker=markers[i], linestyle=linestyles[i], label=features[i])
                        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                        ax.set_title(str(k) + ' of ' + str(prediction_window))
                        plt.subplots_adjust(right=0.8)
                        plt.show()
    
                    # pluck the logits at the final step
                    outputs = outputs[:, -1, :]
                    # Concatenate the predicted value(s) and the forecast for the next step,
                    # since we don't want to predict the model data, but use the current NWP forecast
                    x_new = torch.cat((outputs,x_model[:,block_size+k]), dim=1)
                    x_new = x_new.unsqueeze(0)
                    # Append to the sequence and continue to predict the next step using the newest prediction and the next NWP forecast step
                    x_tmp = torch.tensor(x[:,:,:].detach().clone().numpy(), dtype=torch.float)
                    x_new_tmp = torch.tensor(x_new[:,:,:].clone().detach().numpy(), dtype=torch.float)
                    x = torch.cat((x_tmp, x_new_tmp), dim=1)
                    x.requires_grad = True
    
                # Want to take the average over each prediction step
                dydx_forecast += dydx_tmp
    
            # Calculate the mean by dividing with the number of prediction steps
            # Append to dydx list
            dydx.append(dydx_forecast/npredict)
    
            # Delete data to save memory
            del x, x_obs, x_model, x_cond, x_new, x_tmp, x_new_tmp
    
    
        if Save_data:
            ofile = data_dir + 'gradients.pickle'
            outfile = open(ofile,'wb')
            save_tuple = (dydx)
            pickle.dump(save_tuple,outfile)
            outfile.close()



    # Convert to an array
    dydx = np.array(dydx)
    # Average over all stations
    dydx_mean = np.nanmean(dydx, axis=0)
    # Average over all stations and prediction steps
    dydx_mean_all = np.nanmean(np.nanmean(dydx, axis=0), axis=0)

 

#    # TEMPORARILY REMOVE THE SIN/COS features
#    features_new = [feature for feature in features if ('cos' not in feature)]
#    features = [feature if 'sin' not in feature else feature[4:] for feature in features_new]
#    Nfeatures = len(features)


    if Plot_dydx_mean_all:
        fig, ax = plt.subplots(figsize=[12,8])
        for i in range(Nfeatures):
            ax.plot(dydx_mean_all[:,i], color=colors[i], marker=markers[i], linestyle=linestyles[i], label=features[i])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=6)
#        ax.set_title(str(k) + ' of ' + str(prediction_window))
        plt.subplots_adjust(right=0.8)
        plt.show()

    if Plot_dydx_mean:
        for k in range(prediction_window):
            fig, ax = plt.subplots(figsize=[12,8])
            for i in range(Nfeatures):
                ax.plot(dydx_mean[k,:,i], color=colors[i], marker=markers[i], linestyle=linestyles[i], label=features[i])
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#            ax.set_title(str(k) + ' of ' + str(prediction_window))
            plt.subplots_adjust(right=0.8)
            plt.show()


    if Plot_dydx:
        k = 0
#        for k in range(prediction_window):
        for istation in range(nstation_test):
            fig, ax = plt.subplots(figsize=[12,8])
            for i in range(Nfeatures):
                ax.plot(dydx[istation][k,:,i], color=colors[i], marker=markers[i], linestyle=linestyles[i], label=features[i])
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=8)
#            ax.set_title('Lead time ' + str(k) + ' of ' + str(prediction_window))
            ax.set_title('Station no. ' + str(istation) + ' of ' + str(nstation_test))
            plt.subplots_adjust(right=0.8)
            plt.show()

