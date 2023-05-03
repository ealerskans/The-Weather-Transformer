"""
NAME:
         Dataset

PURPOSE:
         Definitions for Dataset class and functions

CONTAINS:

"""

#==========================
# Standard library imports
#==========================
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
import pandas as pd
import random
import sys
import time
import torch
from torch.utils.data import Dataset

#===============
# Local imports
#===============
from general import GetDataVariables
from read_weather_data import calculate_wsdir
from tools import convert_unix_times, calculate_day_of_year, calculate_hour_of_day, cyclic_encoding



#================================
# Global, module-wise, variables
#================================
# For creating analysis structured array
analysis_names, analysis_types = GetDataVariables('analysis')
analysis_array_type_list = []
for i in range(len(analysis_names)):
    analysis_array_type_list.append((analysis_names[i], analysis_types[i]))





class ForecastDataset(Dataset):
    """
     Forecast dataset class
    """

    def __init__(self, data, station_idx, block_size,feature_indices):
        self.data = data
        self.station_idx = station_idx
        self.block_size = block_size
        self.feature_indices = feature_indices
        # Create so I don't have to re-create every time I run the __getitem__ method
        self.nstation = np.unique(self.station_idx).shape[0]

        # Get station list
        station_list = np.unique(self.station_idx)
        # Number of data for each station - block_size removed
        obs_len_block_size = [np.sum(self.station_idx == station_list[istation]) - self.block_size for istation in range(self.nstation)]
        Nobs_len_block_size = sum(obs_len_block_size)
        # Cumulative number - block_size removed
        self.obs_cumulative = np.cumsum([0]+np.array(obs_len_block_size))

        # Create a list of station indices to get when supplying an index later on in getitem
        # Check which cumulative number idata is less than to get station index
        self.istation = np.zeros(Nobs_len_block_size, dtype=int)
        for idata in range(Nobs_len_block_size):
            self.istation[idata] = np.argwhere(idata < self.obs_cumulative)[0,0]


    def __len__(self):
        return self.data.shape[0] - self.nstation * self.block_size

    def __getitem__(self, idata):
        # Station index for the data itself
        idx = idata + self.istation[idata]*self.block_size

        # Construct the training chunks
        dix = self.data[idx:idx + self.block_size + 1,self.feature_indices]
        x = torch.tensor(dix[:-1,:], dtype=torch.float)
        y = torch.tensor(dix[1:,:], dtype=torch.float)


        return x, y



def DataPreprocessing(OBSraw,NWPraw,OBSext,NWPext,station_index,features,verbose=False,clean=True):
    """
    Preprocess the data (shift + calculate new variables)

    Input:
    + OBSraw: Raw observational data on form:
              (nstation, ndata) - list of arrays
    + NWPraw: Raw NWP data on form:
              (nstation, nfc, nlead_time) - list of arrays
    + OBSext: Observational data matched to raw NWP data:
              (nstation, nfc, nlead_time) - list of arrays
    + NWPext: NWP data matched to raw station data (newest forecast only):
              (nstation, ndata) - list of arrays
    + station_index: Station index on form:
                 (nstation, ndata) - list of arrays
    + features: List of all features
    + verbose: Logical for deciding if function is verbose or not
    + clean: Logical for deciding if cleaning/deletion variables should be performed

    Output:
    + data_shifted: Data with the newest forecast used - list of arrays
    + extra_data_shifted: Extra (temporal) features with the newest forecast used - list of arrays
    + data_shifted_1d: Data with the newest forecast used - 1D array
    + data_raw: Data on raw NWP format - list of arrays
    + extra_data_raw: Extra (temporal) features with data on the raw NWP format - list of arrays
    + station_index_shifted: Station index for data_shifted - list of arrays
    + station_index_shifted_1d: Station index for data_shifted_1d - 1D array
    """

    # Dimensions 
    Nfeatures = len(features)
    nstation = len(NWPraw)
    nlead_time = NWPraw[0]['t2m'].shape[1]
    ndata = [OBSraw[istation]['pres'].shape[0] for istation in range(nstation)]
    nfc = [NWPraw[istation]['t2m'].shape[0] for istation in range(nstation)]
    Ntot = sum(ndata) - nstation  # Since the data is shifted for each station exclude one point for each station

    # Initializations
    data_shifted = []
    extra_data_shifted = []
    data_raw = []
    extra_data_raw = []
    station_index_shifted = []

    # OBS features
    iobs_features = [i for i,feature in enumerate(features) if 'obs' in feature]
    obs_features = [features[iobs_feature] for iobs_feature in iobs_features]
    obs_data_features = [feature[:-4] for feature in obs_features]
    # NWP features - don't include the features that need to be pre-processed (e.g. wind direction)
    inwp_features = [i for i,feature in enumerate(features) \
                       if ( ( ('nwp' in feature) & ('sin' not in feature) & ('cos' not in feature) ) & \
                            ( ('nwp' in feature) & ('wspd500' not in feature) ) & \
                            ( ('nwp' in feature) & ('wspd700' not in feature) ) & \
                            ( ('nwp' in feature) & ('wspd850' not in feature) ) )]
    nwp_features = [features[inwp_feature] for inwp_feature in inwp_features]
    nwp_data_features = [feature[:-4] for feature in nwp_features]
    # wsdir features
    iwspd_features = [i for i,feature in enumerate(features) \
                         if ( ('nwp' in feature) & ( ('wspd500' in feature) | ('wspd700' in feature) | ('wspd850' in feature) ) )]
    wspd_features = [features[iwspd_feature] for iwspd_feature in iwspd_features]
    wspd_data_features = [feature[:-4] for feature in wspd_features]


    for istation in range(nstation):
        if verbose:
            print('Working on station ' + str(istation+1) + ' out of ' + str(nstation))

        # Initializations
        data_shifted.append(np.full((ndata[istation]-1,Nfeatures), fill_value=np.nan))
        extra_data_shifted.append(np.full((ndata[istation]-1), dtype=analysis_array_type_list, fill_value=np.nan))
        data_raw.append(np.full((nfc[istation],nlead_time,Nfeatures), fill_value=np.nan))
        extra_data_raw.append(np.full((nfc[istation],nlead_time), dtype=analysis_array_type_list, fill_value=np.nan))
        station_index_shifted.append(station_index[istation][:-1])

        # Get features:
        #--------------
        # - OBS features
        # Don't include the last data point since we're shifting the NWP data and we should match the OBS data length to this
        obs_data = structured_to_unstructured(OBSraw[istation][obs_data_features][:-1])
        data_shifted[istation][:,iobs_features] = obs_data
        # Include the last data point since we're NOT shifting the NWP data here and we should match the station data length to this
        obs_data = structured_to_unstructured(OBSext[istation][obs_data_features][:,:])
        data_raw[istation][:,:,iobs_features] = obs_data

        del obs_data


        # - NWP features
        # Want to shift the forecast data one step to the left, i.e. so that the current obs sees the prediction +1 step ahead
        # Don't want to include the features for which we have the cyclic variables (wsdir) and other variables that need to be pre-processed/calculated
        # Don't include the last data point since this is the first data point (due to the shifting/rolling)
        nwp_input = np.roll(structured_to_unstructured(NWPext[istation][nwp_data_features]), shift=-1, axis=0)
        data_shifted[istation][:,inwp_features] = nwp_input[:][:-1]
        # Include the last data point since we're NOT shifting the NWP data here
        nwp_input = structured_to_unstructured(NWPraw[istation][nwp_data_features][:,:])
        data_raw[istation][:,:,inwp_features] = nwp_input

        del nwp_input


        # Preprocess 10m wind direction
        if ( ('sin_wsdir10m_nwp' in features) & ('cos_wsdir10m_nwp' in features) ):
            if verbose:
                print('Cyclic encoding of 10m wind direction')
            # Maximum of data to be cyclically encoded
            max_val = 360.
            # Get the ifeatures corresponding to wsdir10m
            iwsdir_features = [i for i,feature in enumerate(features) if 'wsdir10m' in feature]
            # - Raw NWP data
            sin_wsdir_raw, cos_wsdir_raw = cyclic_encoding(NWPraw[istation]['wsdir10m'].squeeze(),max_val)

            # - "Extended" NWP data (on OBS form)
            sin_wsdir_ext, cos_wsdir_ext = cyclic_encoding(NWPext[istation]['wsdir10m'].squeeze(),max_val)

            # Don't include the last data point since we're shifting the NWP data and we should match the OBS data length to this
            nwp_input = np.roll(sin_wsdir_ext, shift=-1, axis=0)
            data_shifted[istation][:,iwsdir_features[0]] = nwp_input[:-1]
            nwp_input = np.roll(cos_wsdir_ext, shift=-1, axis=0)
            data_shifted[istation][:,iwsdir_features[1]] = nwp_input[:-1]
            # Include the last data point since we're NOT shifting the NWP data here and we should match the OBS data length to this
            nwp_data = sin_wsdir_raw[:,:]
            data_raw[istation][:,:,iwsdir_features[0]] = nwp_data
            nwp_data = cos_wsdir_raw[:,:]
            data_raw[istation][:,:,iwsdir_features[1]] = nwp_data

            del sin_wsdir_raw, cos_wsdir_raw, sin_wsdir_ext, cos_wsdir_ext


        # Preprocess winds
        for ifeature,feature in enumerate(wspd_data_features):
            if verbose:
                print('Calculation of wind speed and cyclic encoding of wind direction: ' + feature)
            # Maximum of data to be cyclically encoded
            max_val = 360.
            # Get the ifeatures corresponding to the current feature
            iwspd_feature = iwspd_features[ifeature]
            # Get the ifeatures corresponding to sin and cos of the current feature
            sin_wsdir_feature = 'sin_wsdir' + feature[-3:]
            cos_wsdir_feature = 'cos_wsdir' + feature[-3:]
            # Get the ifeatures corresponding to sin and cos of the current feature
            isin_wsdir_feature = [i for i,f in enumerate(features) if sin_wsdir_feature in f][0]
            icos_wsdir_feature = [i for i,f in enumerate(features) if cos_wsdir_feature in f][0]
            # Get the corresponding u and v features
            ufeature = 'u' + feature[-3:]
            vfeature = 'v' + feature[-3:]
            # - Raw NWP data
            wspd_raw = np.sqrt(NWPraw[istation][ufeature].squeeze()**2 + NWPraw[istation][vfeature].squeeze()**2)
            # - "Extended" NWP data (on OBS form)
            wspd_ext = np.sqrt(NWPext[istation][ufeature].squeeze()**2 + NWPext[istation][vfeature].squeeze()**2)
            # - Raw NWP data
            wsdir = calculate_wsdir(NWPraw[istation][ufeature].squeeze(),NWPraw[istation][vfeature].squeeze())
            sin_wsdir_raw, cos_wsdir_raw = cyclic_encoding(wsdir,max_val)
            # - "Extended" NWP data (on OBS form)
            wsdir = calculate_wsdir(NWPext[istation][ufeature].squeeze(),NWPext[istation][vfeature].squeeze())
            sin_wsdir_ext, cos_wsdir_ext = cyclic_encoding(wsdir,max_val)

            # - Wind speed
            # Don't include the last data point since we're shifting the NWP data and we should match the OBS data length to this
            nwp_input = np.roll(wspd_ext, shift=-1, axis=0)
            data_shifted[istation][:,iwspd_feature] = nwp_input[:-1]
            # Include the last data point since we're NOT shifting the NWP data here and we should match the OBS data length to this
            nwp_data = wspd_raw[:,:]
            data_raw[istation][:,:,iwspd_feature] = nwp_data

            # - Wind direction
            # Don't include the last data point since we're shifting the NWP data and we should match the OBS data length to this
            nwp_input = np.roll(sin_wsdir_ext, shift=-1, axis=0)
            data_shifted[istation][:,isin_wsdir_feature] = nwp_input[:-1]
            nwp_input = np.roll(cos_wsdir_ext, shift=-1, axis=0)
            data_shifted[istation][:,icos_wsdir_feature] = nwp_input[:-1]
            # Include the last data point since we're NOT shifting the NWP data here and we should match the OBS data length to this
            nwp_data = sin_wsdir_raw[:,:]
            data_raw[istation][:,:,isin_wsdir_feature] = nwp_data
            nwp_data = cos_wsdir_raw[:,:]
            data_raw[istation][:,:,icos_wsdir_feature] = nwp_data

            del wspd_raw, wspd_ext, wsdir, sin_wsdir_raw, cos_wsdir_raw, sin_wsdir_ext, cos_wsdir_ext


        # Preprocess temporal data
        if ( ('sin_doy_time' in features) & ('cos_doy_time' in features) ):
            if verbose:
                print('Cyclic encoding of day of year')
            # Maximum of data to be cyclically encoded
            max_val = 366
            # Get the ifeatures corresponding to doy
            idoy_features = [i for i,feature in enumerate(features) if 'doy' in feature]
            # Get day or year from the epoch time - convert to datetime and then get day of year
            # - Raw OBS data
            doy_raw = np.expand_dims(calculate_day_of_year(OBSraw[istation]['unix']), axis=1)
            sin_doy_raw, cos_doy_raw = cyclic_encoding(doy_raw,max_val)   # Include leap years!

            # - Extended OBS data
            doy_ext = np.expand_dims(calculate_day_of_year(OBSext[istation]['unix']), axis=2)
            sin_doy_ext, cos_doy_ext = cyclic_encoding(doy_ext,max_val)   # Include leap years!

            # Don't include the last data point since we're shifting the NWP data and we should match the OBS data length to this
            time_data = sin_doy_raw[:-1,:]
            data_shifted[istation][:,idoy_features[0]] = time_data.squeeze()
            time_data = cos_doy_raw[:-1,:]
            data_shifted[istation][:,idoy_features[1]] = time_data.squeeze()
            # Include the last data point since we're NOT shifting the NWP data here and we should match the OBS data length to this
            time_data = sin_doy_ext[:,:]
            data_raw[istation][:,:,idoy_features[0]] = time_data.squeeze()
            time_data = cos_doy_ext[:,:]
            data_raw[istation][:,:,idoy_features[1]] = time_data.squeeze()

            del doy_raw, doy_ext, sin_doy_raw, cos_doy_raw, sin_doy_ext, cos_doy_ext, time_data


        # Preprocess temporal data
        if ( ('sin_hod_time' in features) & ('cos_hod_time' in features) ):
            if verbose:
                print('Cyclic encoding of hour of day')
            # Maximum of data to be cyclically encoded
            max_val = 23
            # Get the ifeatures corresponding to hod
            ihod_features = [i for i,feature in enumerate(features) if 'hod' in feature]
            # Get hour of day from the epoch time - convert to datetime and then get hour of day
            # - Raw OBS data
            hod_raw = np.expand_dims(calculate_hour_of_day(OBSraw[istation]['unix']), axis=1)
            sin_hod_raw, cos_hod_raw = cyclic_encoding(hod_raw,max_val)

            # - Extended OBS data
            hod_ext = np.expand_dims(calculate_hour_of_day(OBSext[istation]['unix']), axis=2)
            sin_hod_ext, cos_hod_ext = cyclic_encoding(hod_ext,max_val)

            # - Temporal features
            # Don't include the last data point since we're shifting the NWP data and we should match the OBS data length to this
            time_data = sin_hod_raw[:-1,:]
            data_shifted[istation][:,ihod_features[0]] = time_data.squeeze()
            time_data = cos_hod_raw[:-1,:]
            data_shifted[istation][:,ihod_features[1]] = time_data.squeeze()
            # Include the last data point since we're NOT shifting the NWP data here and we should match the OBS data length to this
            time_data = sin_hod_ext[:,:]
            data_raw[istation][:,:,ihod_features[0]] = time_data.squeeze()
            time_data = cos_hod_ext[:,:]
            data_raw[istation][:,:,ihod_features[1]] = time_data.squeeze()

            del hod_raw, hod_ext, sin_hod_raw, cos_hod_raw, sin_hod_ext, cos_hod_ext, time_data



        # Extra data_raw
        nwp_input = np.roll(NWPext[istation]['issue_unix'], shift=-1, axis=0)
        extra_data_shifted[istation]['issue_unix'] = nwp_input[:-1]
        nwp_input = np.roll(NWPext[istation]['valid_unix'], shift=-1, axis=0)
        extra_data_shifted[istation]['valid_unix'] = nwp_input[:-1]

        del nwp_input


        # Extra data_raw
        extra_data_raw[istation]['issue_unix'] = NWPraw[istation]['issue_unix'][:,:]
        extra_data_raw[istation]['valid_unix'] = NWPraw[istation]['valid_unix'][:,:]


        if clean:
            # Clean up under the input data since otherwise the program will get killed
            OBSraw[istation] = []
            OBSext[istation] = []
            NWPraw[istation] = []
            NWPext[istation] = []


    # Initializations 1D array
    data_shifted_1d = np.full((Ntot,Nfeatures), fill_value=np.nan)
    for ifeature,feature in enumerate(features):
        data = [data_shifted[istation][:,ifeature] for istation in range(nstation)]
        data_shifted_1d[:,ifeature] = np.hstack(data).squeeze()#np.column_stack(data)

    station_index_shifted_1d = np.hstack(station_index_shifted).squeeze()


    return OBSraw, NWPraw, OBSext, NWPext, data_shifted, extra_data_shifted, data_shifted_1d, \
           data_raw, extra_data_raw, station_index_shifted, station_index_shifted_1d



def TrainValidationTestSplit(imei,nstation_train,nstation_val,nstation_test,data_shifted,extra_data_shifted,data_shifted_1d, \
                             data_raw,extra_data_raw,station_index,station_index_1d,verbose=False,clean=False):
    """
    Define the train, validation and test datasets

    Input:
    + imei: Station IMEI number
    + nstation_val: Number of stations in validation dataset
    + nstation_test: Number of stations in test dataset
    + data_shifted: Dataset where only the newest forecast is used
    + extra_data_shifted: Dataset where only the newest forecast is used - extra features!
    + data_shifted_1d: data_shifted on 1d array format
    + data_raw: Dataset on raw NWP format
    + extra_data_raw: Extra variables on raw NWP format
    + station_index: Station index for each of the stations/locations
    + station_index_1d: Station index on 1d array format
    + verbose: How verbose the functions should be about what it's doing

    Output:
    + data_train: Training data - list of arrays
    + extra_data_train: Extra features for the training data - list of arrays
    + data_train_1d: Training data - array
    + data_raw_train: Training data on raw NWP format
    + extra_data_raw_train: Extra features for training data - on raw NWP format
    + station_train_index: Station index for data_train dataset
    + station_train_index_1d: Station index for data_train_1d dataset
    + data_val: Validation data - list of arrays
    + extra_data_val: Validation data - list of arrays
    + data_val: Validation data - array
    + data_raw_val: Validation data on raw NWP form
    + extra_data_raw_val: Extra features for Validation data - on raw NWP format
    + station_val_index: Station index for data_Validation dataset
    + station_val_index_1d: Station index for data_Validation dataset
    + data_test: Test data - list of arrays
    + extra_data_test Test data - list of arrays
    + data_test1d: Test data - array
    + data_raw_test Test data on raw NWP form
    + extra_data_raw_test Extra features for test data - on raw NWP format
    + station_test_index: Station index for data_test dataset
    + station_test_index_1d: Station index for data_test_1d dataset
    + station_train: Station indices for training dataset
    + station_val: Station indices for validation dataset
    + station_test: Station indices for test dataset
    """
    if verbose:
        print('Train/validation/test split')

    # Dimensions
    nstation = len(data_shifted)

    # Array of station indices
    station_tot = np.arange(nstation)

    # Randomly select the stations to be used for validation
    station_idx = np.array(nstation)
    istation_val = np.random.choice(station_idx,nstation_val,replace=False)
    station_val = station_tot[istation_val]
    station_tot = np.delete(station_tot,istation_val)
    # Randomly select the stations to be used for test
    nstation_tot = len(station_tot)
    station_idx = np.array(nstation_tot)
    istation_test = np.random.choice(station_idx,nstation_test,replace=False)
    station_test = station_tot[istation_test]
    station_tot = np.delete(station_tot,istation_test)
    # Use the data from the other stations for training
    station_train = station_tot



    # Specify the subsets for the training, validation and test datasets
    data_train = [data_shifted[istation][:,:] for istation in station_train]
    data_val = [data_shifted[istation][:,:] for istation in station_val]
    data_test = [data_shifted[istation][:,:] for istation in station_test]
    if clean:
        data_shifted[:] = []

    extra_data_train = [extra_data_shifted[istation][:] for istation in station_train]
    extra_data_val = [extra_data_shifted[istation][:] for istation in station_val]
    extra_data_test = [extra_data_shifted[istation][:] for istation in station_test]
    if clean:
        extra_data_shifted[:] = []

    data_raw_train = [data_raw[istation][:,:,:] for istation in station_train]
    data_raw_val = [data_raw[istation][:,:,:] for istation in station_val]
    data_raw_test = [data_raw[istation][:,:,:] for istation in station_test]
    if clean:
        data_raw[:] = []

    extra_data_raw_train = [extra_data_raw[istation][:][:,:] for istation in station_train]
    extra_data_raw_val = [extra_data_raw[istation][:][:,:] for istation in station_val]
    extra_data_raw_test = [extra_data_raw[istation][:][:,:] for istation in station_test]
    if clean:
        extra_data_raw[:] = []

    station_train_index = [station_index[istation] - min(station_train) for istation in station_train]
    station_val_index = [station_index[istation] - min(station_val) for istation in station_val]
    station_test_index = [station_index[istation] - min(station_test) for istation in station_test]
    if clean:
        station_index[:] = []

    data_train_1d = data_shifted_1d[np.in1d(station_index_1d,station_train),:]
    data_val_1d = data_shifted_1d[np.in1d(station_index_1d,station_val),:]
    data_test_1d = data_shifted_1d[np.in1d(station_index_1d,station_test),:]
    if clean:
        data_shifted_1d[:] = 0.

    station_train_index_1d = station_index_1d[np.in1d(station_index_1d,station_train)] - min(station_train)
    station_val_index_1d = station_index_1d[np.in1d(station_index_1d,station_val)] - min(station_val)
    station_test_index_1d = station_index_1d[np.in1d(station_index_1d,station_test)] - min(station_test)
    if clean:
        station_index_1d[:] = 0.


    return data_shifted, extra_data_shifted, data_raw, extra_data_raw, station_index, station_index_1d, data_shifted_1d, \
           data_train, extra_data_train, data_train_1d, data_raw_train, extra_data_raw_train, station_train_index, station_train_index_1d, \
           data_val, extra_data_val, data_val_1d, data_raw_val, extra_data_raw_val, station_val_index, station_val_index_1d, \
           data_test, extra_data_test, data_test_1d, data_raw_test, extra_data_raw_test, station_test_index, station_test_index_1d, \
           station_train, station_val, station_test



def DataNormalization(data_train,data_raw_train,data_train_1d,data_val,data_raw_val,data_val_1d, \
                      data_test,data_raw_test,data_test_1d,Nfeatures,features,verbose):
    """
    Normalize the features based on the training data ONLY

    Input:
    + data_train: Training data
    + data_raw_train: Training data on raw NWP form
    + data_val: Validation data
    + data_raw_val: Validation data on raw NWP form
    + data_test: Test data with only the newest forecast
    + data_test: Test data on raw NWP form
    + Nfeatures: Number of features in the data array

    Output:
    + data_train: Normalized data_train
    + data_raw_train: Normalized data_raw_train
    + data_val: Normalized data_val
    + data_raw_val: Normalized data_raw_val
    + data_test: Normalized data_test
    + data_raw_test: Normalized data_raw_test
    + mu: Mean for each feature
    + std: Standard deviation for each feature
    """
    if verbose:
        print('Data normalization')

    # Dimensions
    nstation_train = len(data_train)
    nstation_val = len(data_val)
    nstation_test = len(data_test)

    # Normalize the data with mean and std
    std = np.full(Nfeatures, fill_value=np.nan)
    mu = np.full(Nfeatures, fill_value=np.nan)

    # Cyclic features, which should not be normalized
    cyclic_features = ['sin_doy_time','cos_doy_time','sin_wsdir_nwp','cos_wsdir_nwp']
    cyclic_features = [feature for i,feature in enumerate(features) if ( ('sin' in feature) | ('cos' in feature) )]

    for ifeature,feature in enumerate(features):
        # Calculate mean and stdev
        if feature in cyclic_features:
            mu[ifeature] = 0.
            std[ifeature] = 1.
        else:
            mu[ifeature] = np.nanmean(data_train_1d[:,ifeature].flatten())
            std[ifeature] = np.nanstd(data_train_1d[:,ifeature].flatten())


    # Normalize all 3 datasets
    print('  - Training dataset')
    data_train_1d = ( data_train_1d - mu )/std
    for istation in range(nstation_train):
        data_train[istation] = ( data_train[istation] - mu )/std
        data_raw_train[istation] = ( data_raw_train[istation] - mu )/std
    print('  - Validation dataset')
    data_val_1d = ( data_val_1d - mu )/std
    for istation in range(nstation_val):
        data_val[istation] = ( data_val[istation] - mu )/std
        data_raw_val[istation] = ( data_raw_val[istation] - mu )/std
    print('  - Test dataset')
    data_test_1d = ( data_test_1d - mu )/std
    for istation in range(nstation_test):
        data_test[istation] = ( data_test[istation] - mu )/std
        data_raw_test[istation] = ( data_raw_test[istation] - mu )/std


    return data_train, data_raw_train, data_train_1d, data_val, data_raw_val, data_val_1d, \
           data_test, data_raw_test, data_test_1d, mu, std




def TestDataset(data,data_raw,extra_data_raw,data_1d,feature_indices,nfc_input,fc_update, \
                      fc_output_interval,prediction_window,block_size,station_test,verbose):
    """
    Create the test dataset
    """
    if verbose:
        print('Create test dataset')

    # Dimensions
    nstation = len(data)   # Number of stations
    ndata = [data[istation].shape[0] for istation in range(nstation)]  # Number of data for each station
    nfeatures = len(feature_indices)#data[0].shape[1]
    nfc = [data_raw[istation].shape[0] for istation in range(nstation)]
    nlead_time = data_raw[0].shape[1]


    # Can't predict on the first block_size data (for each station) since this we're using as input for the
    # first prediction
    # Number of prediction cases - all values in between the current and next forecast to predict
    npredict = [(ifc - nfc_input)*int(fc_update/fc_output_interval) for ifc in nfc]   

    # Features
    ifeatures = np.arange(nfeatures)
    obs_features = feature_indices[0]
    nwp_features = feature_indices[1:]
    iobs_features = ifeatures[0]
    inwp_features = ifeatures[1:]

    # Loop over stations in order to initialize the test dataset array for each station
    test_dataset = []
    analysis_dataset = []
    extra_analysis_dataset = []
    for istation in range(nstation):
        # Initialize the test datasets
        test_dataset_tmp = np.full((npredict[istation],block_size+prediction_window,nfeatures), fill_value=np.nan)
        analysis_dataset_tmp = np.full((npredict[istation],prediction_window,2), fill_value=np.nan)

        # Initialize array with extra features
        extra_analysis_dataset_tmp = np.full((npredict[istation],prediction_window), fill_value=np.nan, dtype=analysis_array_type_list)

        # Loop over number of prediction data points
        i0 = 0
        for ipredict in range(npredict[istation]):
            # (1). Construct the test dataset for predictions
            # Temporary array to store the data in - need to add 1 to account for the fact that I'll be rolling the array later on!
            data_tmp = np.zeros((prediction_window,nfeatures))

            # Indices for the data with the newest forecast
            i1 = i0 + block_size
            # Indices for the data with the current forecast
            if ipredict % 2 == 0:
                ifc = nfc_input + ipredict//2
                ilead = 0

            # Shifted (using newest forecast only)
            data_shifted = data[istation][i0:i1,feature_indices]
            # Unshifted (data on raw NWP form) - add one data point since I'm going to be rolling the array later on!
            data_unshifted = data_raw[istation][ifc,ilead:ilead+prediction_window+1,:]

            # Need to shift the unshifted data since in the prediction we're using the forecast for
            # +1 step ahead and the current observation to predict the +1 step ahead observation
            # - OBS data
            # Observed variable (temperature) - use unshifted data
            data_tmp[:,iobs_features] = data_unshifted[:prediction_window,obs_features]
            # - NWP data
            # Want to shift the forecast data one step to the left, i.e. so that the current obs sees the prediction +1 step ahead
            # Don't include the last data point since this is the first data point
            nwp_input = np.roll(data_unshifted[:,nwp_features], shift=-1, axis=0)
            # Include the last data point - will exclude it later
            data_tmp[:,inwp_features] = nwp_input[:prediction_window,:]

            # Combine the datasets
            test_dataset_tmp[ipredict,:,:] = np.concatenate((data_shifted, data_tmp), axis=0)

            # (2). Construct the dataset used for testing our predictions and for plotting 
            analysis_dataset_tmp[ipredict,:,0] = data_raw[istation][ifc,ilead:ilead+prediction_window,0]
            analysis_dataset_tmp[ipredict,:,1] = data_raw[istation][ifc,ilead:ilead+prediction_window,1]
            extra_analysis_dataset_tmp['issue_unix'][ipredict,:] = extra_data_raw[istation]['issue_unix'][ifc,ilead:ilead+prediction_window]
            extra_analysis_dataset_tmp['valid_unix'][ipredict,:] = extra_data_raw[istation]['valid_unix'][ifc,ilead:ilead+prediction_window]

            # Advance i0
            # NOTE: If I advance by 2, then I only predict on the first value in the forecast
            #       I want to predict using both the first and the second value, since for the third value
            #       I would have gotten a new forecast and will therefore be using that instead
            # NOTE: If I use both the frst and the second value then I need to redefine how I get ifc
            #       Since now it's based on the +2 of i0
            # NOTE: This is valid for 3-HOURLY output!!!
            i0 += 1
            ilead += 1

        # Convert to a tesnor
        test_dataset_tmp = torch.tensor(test_dataset_tmp, dtype=torch.float)
        analysis_dataset_tmp = torch.tensor(analysis_dataset_tmp, dtype=torch.float)

        # Append the data
        test_dataset.append(test_dataset_tmp)
        analysis_dataset.append(analysis_dataset_tmp)
        extra_analysis_dataset.append(extra_analysis_dataset_tmp)


    return test_dataset, analysis_dataset, extra_analysis_dataset
