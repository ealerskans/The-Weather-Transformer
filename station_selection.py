"""
Construct the dataset used for training, validation and test.
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
import numpy as np
import os
import pandas as pd
import sys
import time

#===============
# Local imports
#===============
from Dataset import DataPreprocessing, DataNormalization, TrainValidationTestSplit
from general import GetDataVariables, SpecifyFeatures, SpecifyDatasetFile
from global_parameters import *
from read_weather_data import ReadStationOBS, ReadStationGFS, save_NWP_OBS_hdf5, \
                              save_training_datasets, save_test_datasets, \
                              save_val_datasets, save_auxiliary_data
from tools import convert_unix_times




if __name__ == '__main__':
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


    # Put the station latitude and longitude in a separate list with only one value for each station
    station_lat = latitudes.tolist()
    station_lon = longitudes.tolist()


    # Get dates in datetime format
    start_datetimes = [datetime.datetime.strptime(start_date,fmt_strp) for start_date in start_dates]
    end_datetimes = [datetime.datetime.strptime(end_date,fmt_strp) for end_date in end_dates]


    # Logicals for saving
    Save_GFS_raw = True
    Save_OBS_raw = True
    Save_GFS_raw_matched = True
    Save_OBS_raw_matched = True
    Save_GFS_data = True
    Save_OBS_data = True
    Save_Dataset = True


    # Feature inclusion - include all in the dataset
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
    include_features['TIME_sin_doy'] = True
    include_features['TIME_cos_doy'] = True
    include_features['TIME_sin_hod'] = True
    include_features['TIME_cos_hod'] = True
    include_features['NWP_wspd500'] = True
    include_features['NWP_wspd500'] = True
    include_features['NWP_wspd700'] = True
    include_features['NWP_wspd700'] = True
    include_features['NWP_wspd850'] = True
    include_features['NWP_wspd850'] = True
    include_features['NWP_sin_wsdir500'] = True
    include_features['NWP_cos_wsdir500'] = True
    include_features['NWP_sin_wsdir700'] = True
    include_features['NWP_cos_wsdir700'] = True
    include_features['NWP_sin_wsdir850'] = True
    include_features['NWP_cos_wsdir850'] = True

    # Specify features
    features, Nfeatures = SpecifyFeatures(include_features)


    # Get NWP and OBS variables
    nwp_names, nwp_types = GetDataVariables('nwp')
    obs_names, obs_types = GetDataVariables('obs')



    print('Read GFS data')
    nwp_files = np.zeros(nstation, dtype=object)
    nfc = np.zeros(nstation, dtype=int)
    for istation in range(nstation):
        # Dates for input file
        start_date = datetime.datetime.strptime(start_dates[istation], fmt_strp).strftime(fmt_out)
        end_date = datetime.datetime.strptime(end_dates[istation], fmt_strp).strftime(fmt_out)
        # The GFS data is saved on the format: days x nfc_cycles (4) x nlead_times
        # nfc = days * nfc_cycles
        nwp_files[istation] = nwp_data_dir + 'GFS_' + str(imei[istation]) + '_' + start_date + '_' + end_date + '.nc'
    
        # Number of foercasts to loop over - individual for each location
        start_datetime = datetime.datetime.strptime(start_date,fmt_out)
        end_datetime = datetime.datetime.strptime(end_date,fmt_out)
        # Need to change the start datetime since I want to have data for the whole forecast
        start_datetime = datetime.datetime(start_datetime.year,start_datetime.month,start_datetime.day,0,0,0)
        # Need to change the end datetime since we want to have data for all of the last day
        end_datetime = datetime.datetime(end_datetime.year,end_datetime.month,end_datetime.day,0,0,0) \
                                            + datetime.timedelta(days=1)
        diff_day = (end_datetime - start_datetime).total_seconds() / 60 / 60 / 24
        # Round up difference in days to get the number of files
        ndays = (np.ceil(diff_day) + 1).astype(int)
        # Get number of forecasts by multiplying with nfc_cycles
        nfc[istation] = ndays * nfc_cycles
    
    t0 = time.time()
    GFSdata_raw = ReadStationGFS(nstation,nfc,nlead_time,nwp_names,nwp_types,nwp_files)
    print('---Elapsed time: ', time.time()-t0, ' s.')
    
    if Save_GFS_raw:
        print('Save raw GFS data')
        input_file = data_dir + 'GFSdata_raw.hdf5'
        save_NWP_OBS_hdf5(input_file,'GFSdata_raw',GFSdata_raw)

        print('Finished saving raw GFS data')
     


     
    print('Read OBS data')
    station_files = np.zeros(nstation, dtype=object)
    for istation in range(nstation):
        # Dates for input file
        start_date = datetime.datetime.strptime(start_dates[istation], fmt_strp).strftime(fmt_out)
        end_date = datetime.datetime.strptime(end_dates[istation], fmt_strp).strftime(fmt_out)
        station_files[istation] = station_data_dir + 'obs_' + str(imei[istation]) + '_' + start_date + '_' + end_date + '.nc'
 
    t0 = time.time()
    OBSdata_raw = ReadStationOBS(nstation,obs_names,obs_types,station_files)
    print('---Elapsed time: ', time.time()-t0, ' s.')
 
    # Number of obs for each station
    obs_len = [OBSdata_raw[istation].shape[0] for istation in range(nstation)]
 
    if Save_OBS_raw:
        print('Save raw OBS data')
        input_file = data_dir + 'OBSdata_raw.hdf5'
        save_NWP_OBS_hdf5(input_file,'OBSdata_raw',OBSdata_raw)

        print('Finished saving raw OBS data')
    
     
     
    print('Match GFS data')
    t0 = time.time()
    for istation in range(nstation):
        # Get datetimes
        obs_valid_datetime = convert_unix_times(OBSdata_raw[istation]['unix'])[1]
        gfs_valid_datetime = convert_unix_times(GFSdata_raw[istation]['valid_unix'])[1]

        # OBS datetimes
        start_datetime = obs_valid_datetime[0]
        end_datetime = obs_valid_datetime[-1]
        mask = ( (start_datetime <= gfs_valid_datetime) & (gfs_valid_datetime <= end_datetime) )

        # If not all elements are true, discard the forcast
        idx = np.array([np.all(mask[ifc,:]) for ifc in range(nfc[istation])])
        mask[idx,:] = True
        mask[~idx,:] = False

        # Get the number of forecasts
        nfc[istation] = int(np.sum(mask)/nlead_time)

        # Insert the new, matched data
        GFSdata_raw[istation] = GFSdata_raw[istation][:][mask].reshape((nfc[istation],nlead_time))
    
    print('---Elapsed time: ', time.time()-t0, ' s.')

        
    print('Match OBS data')
    t0 = time.time()
    for istation in range(nstation):
        # Get datetimes
        obs_valid_datetime = convert_unix_times(OBSdata_raw[istation]['unix'])[1]
        gfs_valid_datetime = convert_unix_times(GFSdata_raw[istation]['valid_unix'])[1]

        # NWP datetimes
        start_datetime = gfs_valid_datetime[0,0]
        end_datetime = gfs_valid_datetime[-1,-1]
        mask = ( (start_datetime <= obs_valid_datetime) & (obs_valid_datetime <= end_datetime) )

        # Get number of observations
        obs_len[istation] = np.sum(mask)
        OBSdata_raw[istation] = OBSdata_raw[istation][:][mask]
    
    print('---Elapsed time: ', time.time()-t0, ' s.')
        
    # Need to save the matched data as well
    if Save_GFS_raw_matched:
        print('Save matched raw GFS data')
        input_file = data_dir + 'GFSdata_raw_matched.hdf5'
        save_NWP_OBS_hdf5(input_file,'GFSdata_raw',GFSdata_raw)

        print('Finished saving matched GFS data')
    
    if Save_OBS_raw_matched:
        print('Save matched raw OBS data')
        input_file = data_dir + 'OBSdata_raw_matched.hdf5'
        save_NWP_OBS_hdf5(input_file,'OBSdata_raw',OBSdata_raw)

        print('Finished saving matched OBS data')
    
     

    # Re-arrange NWP data array so that we get it in the same format as the OBS data
    nwp_array_type_list = []
    for i in range(len(nwp_names)):
        nwp_array_type_list.append((nwp_names[i], nwp_types[i]))

    # Re-arrange OBS data so that we get it in the same format as the NWP data
    # Remove datetime field
    idx = obs_names.index('datetime')
    obs_names_less = obs_names.copy()
    obs_names_less.pop(idx)
    obs_types_less = obs_types.copy()
    obs_types_less.pop(idx)
    obs_array_type_list = []
    for i in range(len(obs_names_less)):
        obs_array_type_list.append((obs_names_less[i], obs_types_less[i]))

    # Initialize arrays
    GFSdata = []
    OBSdata = []
    print('Get GFS data on OBS format and OBS data on GFS format')
    t0 = time.time()
    for istation in range(nstation):
        # Get GFS and OBS valid times
        gfs_valid_time = convert_unix_times(GFSdata_raw[istation]['valid_unix'])[0].T
        obs_valid_time = convert_unix_times(OBSdata_raw[istation]['unix'])[0].T
 
        GFSdata_tmp = np.zeros((obs_len[istation]),dtype=nwp_array_type_list)
        OBSdata_tmp = np.zeros((nfc[istation],nlead_time),dtype=obs_array_type_list)
        for ifc in range(nfc[istation]-1):
            # GFS data
            #----------
            i0 = int(ifc*fc_update/fc_output_interval)
            i1 = i0 + int(fc_update/fc_output_interval)
            iraw0 = 0#1#
            iraw1 = iraw0 + int(fc_update/fc_output_interval)
            for i,name in enumerate(nwp_names):
                GFSdata_tmp[name][i0:i1] = GFSdata_raw[istation][name][ifc,iraw0:iraw1]

            # OBS data
            #----------
            # Get the observations with the same time as for the forecast!
            obsidx = np.searchsorted(obs_valid_time,gfs_valid_time[ifc,:])
            for i,name in enumerate(obs_names_less):
                OBSdata_tmp[name][ifc,:] = OBSdata_raw[istation][name][obsidx]
 
        # GFS data
        #----------
        # Special consideration for the last forecast 
        for i,name in enumerate(nwp_names):
            GFSdata_tmp[name][i1:] = GFSdata_raw[istation][name][nfc[istation]-1,:]

        # Append data
        GFSdata.append(GFSdata_tmp)


        # OBS data
        #---------
        obsidx = np.searchsorted(obs_valid_time,gfs_valid_time[nfc[istation]-1,:])
        for i,name in enumerate(obs_names_less):
            OBSdata_tmp[name][nfc[istation]-1,:] = OBSdata_raw[istation][name][obsidx]

        # Append data
        OBSdata.append(OBSdata_tmp)
 
        del gfs_valid_time, GFSdata_tmp, obs_valid_time, OBSdata_tmp
 
    print('---Elapsed time: ', time.time()-t0, ' s.')

 
    if Save_OBS_data:
        print('Save OBS data on NWP format')
        input_file = data_dir + 'OBSdata.hdf5'
        save_NWP_OBS_hdf5(input_file,'OBSdata',OBSdata)

        print('Finished saving OBS data')


    if Save_GFS_data:
        print('Save GFS data on OBS format')
        input_file = data_dir + 'GFSdata.hdf5'
        save_NWP_OBS_hdf5(input_file,'GFSdata',GFSdata)

        print('Finished saving GFS data')


 
    print('Convert the GFSdata and OBSdata_raw to 1d arrays')
    Ntot = sum(obs_len)
 
    # GFS structured array
    nwp_array_type_list = []
    for i in range(len(nwp_names)):
        nwp_array_type_list.append((nwp_names[i], nwp_types[i]))

    # OBS structured array
    idx = obs_names.index('datetime')
    obs_names_less = obs_names.copy()
    obs_names_less.pop(idx)
    obs_types_less = obs_types.copy()
    obs_types_less.pop(idx)
    obs_array_type_list = []
    for i in range(len(obs_names_less)):
        obs_array_type_list.append((obs_names_less[i], obs_types_less[i]))

    # Create 1d array with station index for all stations
    station_index = [[istation]*obs_len[istation] for istation in range(nstation)]
    station_index_1d = np.hstack(station_index).squeeze()
 


    # Define the datasets and features
    # + data_shifted = Data with the newest forecast only
    # + extra_data_shifted = Data with the newest forecast only - time variables
    # + data_raw = Data on raw NWP form
    # + extra_data_raw = Data on raw NWP form - time variables
    t0 = time.time()
    OBSdata_raw, GFSdata_raw, OBSdata, GFSdata, data_shifted, extra_data_shifted, data_shifted_1d,  \
    data_raw, extra_data_raw, station_index_shifted, station_index_shifted_1d = \
                    DataPreprocessing(OBSdata_raw,GFSdata_raw,OBSdata,GFSdata,station_index,features,verbose=False,clean=True)

    del station_index, station_index_1d


    # Delete OBS and GFS data since it's no longer necessary to have them
    del OBSdata_raw, GFSdata_raw, OBSdata, GFSdata
    print(' ---Elapsed time DataPreprocessing: ', time.time()  - t0, ' s.')


    # Define train, validation and test datasets
    t0 = time.time()
    data_shifted, extra_data_shifted, data_raw, extra_data_raw, station_index_shifted, station_index_shifted_1d, data_shifted_1d, \
    data_train, extra_data_train, data_train_1d, data_raw_train, extra_data_raw_train, station_train_index_shifted, station_train_index_shifted_1d, \
    data_val, extra_data_val, data_val_1d, data_raw_val, extra_data_raw_val, station_val_index_shifted, station_val_index_shifted_1d, \
    data_test, extra_data_test, data_test_1d, data_raw_test, extra_data_raw_test, station_test_index_shifted, station_test_index_shifted_1d, \
    station_train, station_val, station_test = TrainValidationTestSplit(imei,nstation_train,nstation_val,nstation_test,data_shifted,extra_data_shifted, \
                                                            data_shifted_1d,data_raw,extra_data_raw,station_index_shifted,station_index_shifted_1d,\
                                                            verbose=False,clean=True)
    print(' ---Elapsed time TrainTestValidationSplit: ', time.time()  - t0, ' s.')
 
    # Delete data that is no longer needed
    del data_shifted, extra_data_shifted, data_shifted_1d, data_raw, extra_data_raw, station_index_shifted, station_index_shifted_1d



    # The normalization should be based on the training data exclusively
    t0 = time.time()
    data_train, data_raw_train, data_train_1d, data_val, data_raw_val, data_val_1d, \
    data_test, data_raw_test, data_test_1d, mu, std = \
             DataNormalization(data_train,data_raw_train,data_train_1d,data_val,data_raw_val,data_val_1d, \
                               data_test,data_raw_test,data_test_1d,Nfeatures,features,verbose=False)
    print(' ---Elapsed time DataNormalization: ', time.time()  - t0, ' s.')



    if Save_Dataset:
        print('Save Datasets')

        # Delete unnecessary data
        del station_train_index_shifted, station_val_index_shifted, station_test_index_shifted

        # Specify output file names
        dataset_file_generic, dataset_file_train, dataset_file_val, dataset_file_test = \
              SpecifyDatasetFile(data_dir,include_features,nstation_train,nstation_val,nstation_test,input_days)

        print(' ---Training data')
        # ---Save the training data in hdf5 format
        # Create data list and group names
        data_list = [data_train, extra_data_train, data_raw_train, extra_data_raw_train]
        # Delete already-used data
        del data_train, extra_data_train, data_raw_train, extra_data_raw_train
        data_groups = ['data_train', 'extra_data_train', 'data_raw_train', 'extra_data_raw_train']

        # Save training datasets
        save_training_datasets(dataset_file_train,data_list,data_groups,data_train_1d,station_train_index_shifted_1d)

        # Delete already-used data
        del data_list, data_groups, data_train_1d, station_train_index_shifted_1d


        print(' ---Validation data')
        # ---Save the validation data in hdf5 format
        # Create data list and group names
        data_list = [data_val, extra_data_val, data_raw_val, extra_data_raw_val]
        # Delete already-used data
        del data_val, extra_data_val, data_raw_val, extra_data_raw_val
        data_groups = ['data_val', 'extra_data_val', 'data_raw_val', 'extra_data_raw_val']
        
        # Save the validation datasets
        save_val_datasets(dataset_file_val,data_list,data_groups,data_val_1d,station_val_index_shifted_1d)

        # Delete already-used data
        del data_list, data_groups, data_val_1d, station_val_index_shifted_1d


        print(' ---Test data')
        # Save the test data in hdf5 format

        # Save test_dataset separately and then remove since it seems to take too much memory otherwise
        print('   ---Saving test_dataset')
        # Create data list and group names for the rest of the test data
        data_list = [data_test, extra_data_test, data_raw_test, extra_data_raw_test]
        # Delete already-used data
        del data_test, extra_data_test, data_raw_test, extra_data_raw_test
        data_groups = ['data_test', 'extra_data_test', 'data_raw_test', 'extra_data_raw_test']

        # Save test datasets
        save_test_datasets(dataset_file_test,data_list,data_groups,data_test_1d)
        # Delete already-used data
        del data_list, data_groups, data_test_1d


        print(' ---Generic data')
        save_tuple = (mu, std, station_train, station_val, station_test)
        save_auxiliary_data(dataset_file_generic,save_tuple)
        del save_tuple


    print('station_selection done')


