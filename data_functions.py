#==========================
# Standard library imports
#==========================
import warnings
# Silence FutureWarnings (something with my numpy version)
warnings.simplefilter(action='ignore', category=FutureWarning)
import datetime
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
from global_parameters import *
from tools import convert_unix_times, calculate_day_of_year, calculate_hour_of_day, shift, shift2d, roll_odd_data




def Roll_T2m_Data(npredict,prediction_window,t2m_nwp,t2m_obs,t2m_pred,extra_analysis_dataset):
    """
    Roll T2m data - only the odd forecast numbers

    Since the forecast/predictions are made every 3rd hour, the same GFS forecast is updated twice,
    before the next forecast is available. Therefore, for the second update/prediction for each forecast
    the second first lead time is used (in this case +6h since +3h is used for the first update).
    This means that in the output for the second forecast, the data need to be shifed so that the first prediction
    corresponds to the second lead time index since this is what it is. Therefore the data need to be rolled,
    but only for every other forecast (odd ones!)
    """
    # Dimensions
    nstation = len(t2m_obs)

    # Initialize lists
    t2m_nwp_roll, t2m_obs_roll, t2m_pred_roll, day_of_year, hour_of_day, \
    ML_update_time, valid_time, valid_datetime = [], [], [], [], [], [], [], []

    t0 = time.time()
    for istation in range(nstation):
        # NWP T2M
        #---------
        dims = (npredict[istation],prediction_window)
        fill_val = np.nan
        t2m_nwp_roll_tmp = roll_odd_data(dims,t2m_nwp[istation],fill_val)
        # Add to list
        t2m_nwp_roll.append(t2m_nwp_roll_tmp)
        del t2m_nwp_roll_tmp
        # Clean up t2m_nwp
        t2m_nwp[istation] = []

        # Station T2M
        #-------------
        dims = (npredict[istation],prediction_window)
        fill_val = np.nan
        t2m_obs_roll_tmp = roll_odd_data(dims,t2m_obs[istation],fill_val)
        # Add to list
        t2m_obs_roll.append(t2m_obs_roll_tmp)
        del t2m_obs_roll_tmp
        # Clean up t2m_obs
        t2m_obs[istation] = []

        # PRED T2M
        #----------
        dims = (npredict[istation],prediction_window)
        fill_val = np.nan
        t2m_pred_roll_tmp = roll_odd_data(dims,t2m_pred[istation],fill_val)
        # Add to list
        t2m_pred_roll.append(t2m_pred_roll_tmp)
        del t2m_pred_roll_tmp
        # Clean up t2m_pred
        t2m_pred[istation] = []


        # DAY OF YEAR
        #--------------------
        day_of_year_roll_tmp = np.full((npredict[istation],prediction_window), dtype=np.float, fill_value=np.nan)
        # Even array
        day_of_year_even = calculate_day_of_year(extra_analysis_dataset[istation]['valid_unix'][::2,:]).astype(np.float)
        # Odd array
        day_of_year_odd = calculate_day_of_year(extra_analysis_dataset[istation]['valid_unix'][1::2,:]).astype(np.float)
        # Roll the odd array
        fill_val = np.nan
        day_of_year_odd_roll = shift2d(day_of_year_odd, 1, 1, fill_value=fill_val)
        # Add even and rolled odd array to new array
        day_of_year_roll_tmp[::2,:] = day_of_year_even
        day_of_year_roll_tmp[1::2,:] = day_of_year_odd_roll
        # Add to list
        day_of_year.append(day_of_year_roll_tmp)
        del day_of_year_roll_tmp

        # HOUR OF DAY
        #--------------------
        hour_of_day_roll_tmp = np.full((npredict[istation],prediction_window), dtype=np.float, fill_value=np.nan)
        # Even array
        hour_of_day_even = calculate_hour_of_day(extra_analysis_dataset[istation]['valid_unix'][::2,:]).astype(np.float)
        # Odd array
        hour_of_day_odd = calculate_hour_of_day(extra_analysis_dataset[istation]['valid_unix'][1::2,:]).astype(np.float)
        # Roll the odd array
        fill_val = np.nan
        hour_of_day_odd_roll = shift2d(hour_of_day_odd, 1, 1, fill_value=fill_val)
        # Add even and rolled odd array to new array
        hour_of_day_roll_tmp[::2,:] = hour_of_day_even
        hour_of_day_roll_tmp[1::2,:] = hour_of_day_odd_roll
        # Add to list
        hour_of_day.append(hour_of_day_roll_tmp)
        del hour_of_day_roll_tmp


        # ML update time
        #----------------
        dims = (npredict[istation],prediction_window)
        fill_val = np.expand_dims(extra_analysis_dataset[istation]['issue_unix'][1::2,0],1)
        ML_update_time_roll_tmp = roll_odd_data(dims,extra_analysis_dataset[istation]['issue_unix'],fill_val)
        # Add to list
        ML_update_time.append(ML_update_time_roll_tmp)
        del ML_update_time_roll_tmp


        # NWP VALID TIME
        #----------------
        dims = (npredict[istation],prediction_window)
        fill_val = np.expand_dims(extra_analysis_dataset[istation]['valid_unix'][1::2,0] - datetime.timedelta(hours=3).total_seconds(),1)
        valid_time_roll_tmp = roll_odd_data(dims,extra_analysis_dataset[istation]['valid_unix'],fill_val)
        # Add to list
        valid_time.append(valid_time_roll_tmp)
        del valid_time_roll_tmp

        # NWP VALID DATETIME
        #--------------------
        valid_datetime_roll_tmp = np.full((npredict[istation],prediction_window), dtype='datetime64[s]', fill_value=np.nan)
        # Even array
        valid_datetime_even = convert_unix_times(extra_analysis_dataset[istation]['valid_unix'][::2,:])[1]
        # Odd array
        valid_datetime_odd = convert_unix_times(extra_analysis_dataset[istation]['valid_unix'][1::2,:])[1]
        # Roll the odd array
        fill_val = np.expand_dims(np.array([np.datetime64(valid_datetime_odd[ifc,0] - datetime.timedelta(hours=3), 's') \
                                            for ifc in range(valid_datetime_odd.shape[0])]),1)
        valid_datetime_odd_roll = shift2d(valid_datetime_odd, 1, 1, fill_value=fill_val)
        # Add even and rolled odd array to new array
        valid_datetime_roll_tmp[::2,:] = valid_datetime_even
        valid_datetime_roll_tmp[1::2,:] = valid_datetime_odd_roll
        # Add to list
        valid_datetime.append(valid_datetime_roll_tmp)
        del valid_datetime_roll_tmp


    print('Elapsed time: ', time.time()-t0, ' s.')
    # Replace t2m with the rolled versions
    t2m_nwp = t2m_nwp_roll.copy()
    t2m_obs = t2m_obs_roll.copy()
    t2m_pred = t2m_pred_roll.copy()
    del t2m_nwp_roll, t2m_obs_roll, t2m_pred_roll


    return t2m_nwp, t2m_obs, t2m_pred, day_of_year, hour_of_day, ML_update_time, valid_time, valid_datetime



def Roll_NWP_Data(npredict,prediction_window,nfc_input,nwp_names,nwp_types,NWPdata_raw):
    """
    Roll the GFS data - only the odd forecast numbers
    """
    # Dimensions
    nstation = len(NWPdata_raw)

    # Create structured data type
    array_type_list = []
    for i in range(len(nwp_names)):
        array_type_list.append((nwp_names[i],nwp_types[i]))

    # Fill value
    fill_val = np.nan

    # Initialize list
    NWPdata_rolled = []

    t0 = time.time()
    for istation in range(nstation):
        # Initialize array
        rolled_data = np.full((npredict[istation],prediction_window), fill_value=np.nan, dtype=array_type_list)

        # Dimensions
        dims = (npredict[istation],prediction_window)

        # Roll the data
        for name in nwp_names:
            data_tmp = np.full((npredict[istation],prediction_window), fill_value=np.nan)
            data_tmp[::2,:prediction_window] = NWPdata_raw[istation][name][nfc_input:,:prediction_window]
            data_tmp[1::2,:prediction_window] = NWPdata_raw[istation][name][nfc_input:,1:1+prediction_window]
            data = roll_odd_data(dims,data_tmp,fill_val)
            rolled_data[name] = data

        # Add to list
        NWPdata_rolled.append(rolled_data)

        # Now we can clean up NWPdata_raw
        NWPdata_raw[istation] = []


    print('Elapsed time: ', time.time()-t0, ' s.')

    return NWPdata_rolled



def data_list21d(step,t2m_obs,t2m_nwp,t2m_pred,hour_of_day,day_of_year,valid_time,valid_datetime,GFSdata_rolled):
    """
    Convert list of 2d arrays into one 1d array
    """
    # Dimensions
    nstation = len(t2m_obs)

    # Names and types
    rolled_names = ['wspd10m', 'wsdir10m', 'rh2m', 'q2m', 'td2m', 'mslp', 'lhtfl', 'shtfl', \
                    'nswrf', 'nlwrf', 'tcc', 'pwat', 'gh500', 'gh700', 'gh850', 't500', \
                    't700', 't850', 'u500', 'u700', 'u850', 'v500', 'v700', 'v850', 'w700', \
                    'doy', 'hod', 'valid_time', 'valid_datetime']
    rolled_types = ['f8'] * (len(rolled_names) - 1) + ['datetime64[s]']


    # Create structured data type
    array_type_list = []
    for i in range(len(rolled_names)):
        array_type_list.append((rolled_names[i],rolled_types[i]))

    # Fill value
    fill_val = np.nan

    # Initialize list
    GFSdata_1dflat = []

    # Total number of data points
    ntot_nwp = sum([t2m_nwp[istation][::step,:].flatten().shape[0] for istation in range(nstation)])
    ntot_pred = sum([t2m_nwp[istation].flatten().shape[0] for istation in range(nstation)])

    # Initialize temperature arrays
    obs_nwp_t2m = np.full((ntot_nwp), fill_value=np.nan); obs_pred_t2m = np.full((ntot_pred), fill_value=np.nan)
    nwp_t2m = np.full((ntot_nwp), fill_value=np.nan);     pred_t2m = np.full((ntot_pred), fill_value=np.nan)


    # Initialize other data (structured array)
    nwp_1dflat = np.full((ntot_nwp),fill_value=fill_val, dtype=array_type_list)
    pred_1dflat = np.full((ntot_pred),fill_value=fill_val, dtype=array_type_list)


    i0_nwp = 0
    i0_pred = 0
    for istation in range(nstation):
        nfc = t2m_pred[istation].shape[0]
        i1_nwp = i0_nwp + t2m_obs[istation][::step,:].flatten().shape[0]
        i1_pred = i0_pred + t2m_obs[istation].flatten().shape[0]

        # T2m
        obs_nwp_t2m[i0_nwp:i1_nwp] = t2m_obs[istation][::step,:].flatten(); obs_pred_t2m[i0_pred:i1_pred] = t2m_obs[istation].flatten()
        nwp_t2m[i0_nwp:i1_nwp] = t2m_nwp[istation][::step,:].flatten();     pred_t2m[i0_pred:i1_pred] = t2m_pred[istation].flatten()

        # GFS data (+temporal data)
        for name in rolled_names:
            if name == 'hod':
                nwp_1dflat[name][i0_nwp:i1_nwp] = hour_of_day[istation][::step,:].flatten()
                pred_1dflat[name][i0_pred:i1_pred] = hour_of_day[istation].flatten()
            elif name == 'doy':
                nwp_1dflat[name][i0_nwp:i1_nwp] = day_of_year[istation][::step,:].flatten()
                pred_1dflat[name][i0_pred:i1_pred] = day_of_year[istation].flatten()
            elif name == 'valid_time':
                nwp_1dflat[name][i0_nwp:i1_nwp] = valid_time[istation][::step,:].flatten()
                pred_1dflat[name][i0_pred:i1_pred] = valid_time[istation].flatten()
            elif name == 'valid_datetime':
                nwp_1dflat[name][i0_nwp:i1_nwp] = valid_datetime[istation][::step,:].flatten()
                pred_1dflat[name][i0_pred:i1_pred] = valid_datetime[istation].flatten()
            else:
                nwp_1dflat[name][i0_nwp:i1_nwp] = GFSdata_rolled[istation][name][::step,:].flatten()
                pred_1dflat[name][i0_pred:i1_pred] = GFSdata_rolled[istation][name].flatten()

        i0_nwp += t2m_obs[istation][::step,:].flatten().shape[0]
        i0_pred += t2m_obs[istation].flatten().shape[0]


    return obs_nwp_t2m, obs_pred_t2m, nwp_t2m, pred_t2m, nwp_1dflat, pred_1dflat
