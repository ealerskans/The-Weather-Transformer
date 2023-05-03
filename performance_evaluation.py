"""
Evaluate the performance of the model
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
from dateutil.relativedelta import relativedelta
import h5py
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
from scipy import interpolate
import sys
import time
import torch

#===============
# Local imports
#===============
from analysis import *
from data_functions import Roll_T2m_Data, Roll_NWP_Data, data_list21d
from Dataset import *
from general import GetDataVariables, InitializeFeatures, SpecifyFeatures, \
                    SpecifyDatasetFile, SpecifyPredictionFile
from global_parameters import *
from read_weather_data import read_training_datasets, read_val_datasets, read_test_datasets, \
                              read_auxiliary_data, read_NWP_OBS_hdf5
from tools import convert_unix_times, calculate_day_of_year, calculate_hour_of_day, \
                  shift, shift2d, roll_odd_data




if __name__ == '__main__':
    # Start time taking:
    t0_tot = time.time()

    print('T2m performance evaluation')


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

    # Location latitude and longitudes as list
    station_lat = latitudes.tolist()
    station_lon = longitudes.tolist()


    # Get dates in datetime format
    start_datetimes = [datetime.datetime.strptime(start_date,fmt_strp) for start_date in start_dates]
    end_datetimes = [datetime.datetime.strptime(end_date,fmt_strp) for end_date in end_dates]


    # Extra settings
    extra_args = {}


    # Logicals for saving
    Save_Plot = True


    # Logicals for calculations
    Calc_Overall_Statistics = True
    Calc_Monthly_Overall_Statistics = True
    Calc_Seasonal_Overall_Statistics = True
    Calc_Statistics = True
    Calc_Seasonal_Statistics = True
    Calc_Monthly_Statistics = True
    Calc_Pooled_Statistics = True
    Calc_Seasonal_Pooled_Statistics = True
    Calc_Monthly_Pooled_Statistics = True


    # Logicals for plotting
    Plot_station_Location = True
    Plot_Seasonal_Statistics_Map = True
    Plot_Statistics = True
    Plot_Seasonal_Pooled_Statistics = True


    # Feature inclusion - either take command-line argument as input or use the ones specified here
    # See if there are any command-line arguments supplied
    args = sys.argv
    include_features, feature_indices = InitializeFeatures(args)


    # Specify features
    features, Nfeatures = SpecifyFeatures(include_features)
    # Specify dataset file name
    dataset_file_generic, dataset_file_train, dataset_file_test, dataset_file_test = \
             SpecifyDatasetFile(data_dir,include_features,nstation_train,nstation_test,nstation_test,input_days)

    # Specify the predictions file
    predictions_file = SpecifyPredictionFile(data_dir,include_features,nstation_train,nstation_test,nstation_test,n_epochs, \
                                             batch_size,input_days,loss_metrics)

    generic_rolled_data_file = data_dir + 'Rolled_Generic_Data_nstation.' + str(nstation_train) + '.' + str(nstation_test) + '.' + str(nstation_test) + '.pickle'


    # Get NWP and OBS variables
    nwp_names, nwp_types = GetDataVariables('nwp')
    obs_names, obs_types = GetDataVariables('obs')


    print(' ---Generic data')
    mu, std, station_train, station_val, station_test = read_auxiliary_data(dataset_file_generic)

    print('Load test dataset')
    print(' ---Test data')
    read_dict={'data_test':True,'data_test_1d':True,'extra_data_test':True, \
               'data_raw_test':True,'extra_data_raw_test':True}
    data_test, data_test_1d, extra_data_test, data_raw_test, extra_data_raw_test = read_test_datasets(dataset_file_test,read_dict)

    # Create test dataset
    t0 = time.time()
    _, analysis_dataset, extra_analysis_dataset\
              = TestDataset(data_test,data_raw_test,extra_data_raw_test,data_test_1d,feature_indices,nfc_input,fc_update, \
                                  fc_output_interval,prediction_window,block_size,station_test,verbose=True)
    print(' ---Elapsed time ValidationDataset: ', time.time()  - t0, ' s.')
    
    # Delete already-used data
    del data_test, data_test_1d, extra_data_test, data_raw_test, extra_data_raw_test


    print('Load GFS data')
    t0 = time.time()
    input_file = data_dir + 'GFSdata_raw_matched.hdf5'
    GFSdata_raw = read_NWP_OBS_hdf5(input_file)

    # Number of forecasts
    nfc = [GFSdata_raw[istation].shape[0] for istation in range(nstation)]

    print('Only keep the GFS data for the test dataset')
    # Clean up GFSdata_raw so that we only have the stations corresponding to the test dataset
    GFSdata_new = []
    for istation in range(nstation_test):
        GFSdata_new.append(GFSdata_raw[station_test[istation]])
        GFSdata_raw[station_test[istation]] = []
    del GFSdata_raw
    GFSdata_raw = GFSdata_new.copy()
    del GFSdata_new

    # Number of forecasts
    nfc = [GFSdata_raw[istation].shape[0] for istation in range(nstation_test)]

    

    # Get the IMEI number of the train, val and test stations
    imei_train = [imei[idx] for idx in station_train]
    imei_val = [imei[idx] for idx in station_val]
    imei_test = [imei[idx] for idx in station_test]
    # Get the lat,lon of the train, val and test stations
    lat_train = [station_lat[idx] for idx in station_train]
    lon_train = [station_lon[idx] for idx in station_train]
    lat_val = [station_lat[idx] for idx in station_val]
    lon_val = [station_lon[idx] for idx in station_val]
    lat_test = [station_lat[idx] for idx in station_test]
    lon_test = [station_lon[idx] for idx in station_test]



    # Plot the location of the stations
    if Plot_station_Location:
        print('Plot station locations')
        Zoom_DK = True
        dataset_name = 'Train'
        PlotStationDatasetMap(lat_train,lon_train,dataset_name,Zoom_DK,Save_Plot,fig_dir)
        dataset_name = 'Validation'
        PlotStationDatasetMap(lat_val,lon_val,dataset_name,Zoom_DK,Save_Plot,fig_dir)
        dataset_name = 'Test'
        PlotStationDatasetMap(lat_test,lon_test,dataset_name,Zoom_DK,Save_Plot,fig_dir)


    #=========================
    print('Load predictions')
    #=========================
    infile = open(predictions_file,'rb')
    save_tuple = pickle.load(infile)
    infile.close()

    t2m_nwp, t2m_obs, t2m_pred = save_tuple
    del save_tuple

    # Since I predict on all available lead times, i.e. one prediction using lead time +0
    # and one prediction using lead time +3, I need to roll every second entry in the t2m
    # arrays if I want them to represent the correct lead time
    npredict = [(ifc - nfc_input)*int(fc_update/fc_output_interval) for ifc in nfc]
    # Roll the data
    t2m_nwp, t2m_obs, t2m_pred, day_of_year, hour_of_day, ML_update_time, valid_time, valid_datetime = \
                          Roll_T2m_Data(npredict,prediction_window,t2m_nwp,t2m_obs,t2m_pred,extra_analysis_dataset)
    


    if Calc_Overall_Statistics:
        print('Calculate overall statistics')
        stat_gfs_all, stat_pred_all, nobs_gfs_all, nobs_pred_all \
              = CalculateOverallStatistics(ML_update_time,valid_time,t2m_obs,t2m_nwp,t2m_pred)

    if Calc_Seasonal_Overall_Statistics:
        print('Calculate seasonal overall statistics')
        stat_season_gfs_all, stat_season_pred_all, nobs_season_gfs_all, nobs_season_pred_all \
              = CalculateSeasonalOverallStatistics(ML_update_time,valid_time,valid_datetime,t2m_obs,t2m_nwp,t2m_pred)


    if Calc_Statistics:
        print('Calculate lead time statistics')
        stat_gfs, stat_pred, nobs_gfs, nobs_pred \
              = CalculateLeadTimeStatistics(ML_update_time,valid_time,t2m_obs,t2m_nwp,t2m_pred)


    if Calc_Seasonal_Statistics:
        print('Calculate seasonal lead time statistics')
        stat_season_gfs, stat_season_pred, nobs_season_gfs, nobs_season_pred \
              = CalculateSeasonalLeadTimeStatistics(ML_update_time,valid_time,valid_datetime,t2m_obs,t2m_nwp,t2m_pred)


    if Calc_Pooled_Statistics:
        print('Calculate pooled lead time statistics')
        stat_gfs_pool, stat_pred_pool, nobs_gfs_pool, nobs_pred_pool = CalculatePooledLeadTimeStatistics(stat_gfs,stat_pred,nobs_gfs,nobs_pred)


    if Calc_Seasonal_Pooled_Statistics:
        print('Calculate seasonal pooled lead time statistics')
        stat_season_gfs_pool, stat_season_pred_pool, nobs_season_gfs_pool, nobs_season_pred_pool = \
                CalculateSeasonalPooledLeadTimeStatistics(stat_season_gfs,stat_season_pred,nobs_season_gfs,nobs_season_pred)





    if Plot_Seasonal_Statistics_Map:
        PlotStationSeasonalStatisticsMap(stat_season_gfs_all,stat_season_pred_all,lat_test,lon_test,fig_dir,Save_Plot,Zoom_DK=True,Validation_Labels=False)

    if Plot_Statistics:
        x = np.arange(prediction_window) * fc_output_interval + 3 # Since we start at + 3h and not + 0h
        PlotPooledLeadTimeStatistics(x,stat_gfs_pool,stat_pred_pool,Save_Plot,fig_dir,**extra_args)

    if Plot_Seasonal_Pooled_Statistics:
        x = np.arange(prediction_window) * fc_output_interval + 3 # Since we start at + 3h and not + 0h
        PlotSeasonalPooledLeadTimeStatistics(x,stat_season_gfs_pool,stat_season_pred_pool,Save_Plot,fig_dir,**extra_args)


    # Final time taking
    print('\n---Total elapsed time for predictions: ', time.time() - t0_tot, ' s.\n\n')




