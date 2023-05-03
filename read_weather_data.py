"""
NAME:
         read_weather_data

PURPOSE:
         Functions for reading in weather data
         (already-interpolated OBS and NWP)

CONTAINS:
         ReadStationNWP
         ReadStationOBS
         ReadKFOutput

"""

#============================================================================
# Standard library imports
#============================================================================
import datetime as dt
import h5py
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
import pandas as pd
import pickle
import sys
import time

#===============
# Local imports
#===============
from datetools import *
from general import GetDataVariables
from global_parameters import *
from tools import convert_unix_times, nan_helper, consecutive_number_ranges, remove_structured_field_name



# Date formats
fmt_strp = '%Y-%m-%d %H:%M:%S'
fmt_strf = '%Y%m%d'
fmt_data = '%Y-%m-%d %H:%M:%S'



def calculate_wsdir(u,v):
    """
    Calculate the meteorological wind direction

    Arguments:
    ----------
    + u: Zonal wind component
    + v: Meridional wind component


    Returns:
    --------
    + wsdir: Wind direction
    """
    return np.mod(180 + rad2deg*np.arctan2(u,v), 360)



def ReadStationGFS(nstation,nfc,nlead_time,var_names,var_types,all_files):
    """
    Function for reading in GFS data at station location from HDF file format
    """

    # List with only GFS variables
    gfs_names = var_names.copy()
    gfs_names.remove('issue_unix')
    gfs_names.remove('valid_unix')
    gfs_names.remove('imei')
    gfs_names.remove('lat')
    gfs_names.remove('lon')

    # Create structured data type
    array_type_list = []
    for i in range(len(var_names)):
        array_type_list.append((var_names[i],var_types[i]))

    # List of station data
    data = []
    # Loop over number of stations
    for istation in range(nstation):
        print('Working on station number ' + str(istation))
        # Create structured array
        data_tmp = np.zeros((nfc[istation],nlead_time),dtype=array_type_list)

        # Read data from netCDF files
        nc_f = all_files[istation]  # Your filename
        nc_fid = h5py.File(nc_f, 'r')

        # Don't want to use the analysis since not all variables are available (also, it's the analysis)
        # Exclude analysis and use the +3h - +54h lead times (nlead_time = 18)
        ilead = 1
        # Variables
        data_tmp['t2m'][:,:] = np.reshape(nc_fid['t2m'][:,:,ilead:], (nfc[istation],nlead_time))
        data_tmp['rh2m'][:,:] = np.reshape(nc_fid['rh2m'][:,:,ilead:], (nfc[istation],nlead_time))
        data_tmp['q2m'][:,:] = np.reshape(nc_fid['q2m'][:,:,ilead:], (nfc[istation],nlead_time))
        data_tmp['u10m'][:,:] = np.reshape(nc_fid['u10m'][:,:,ilead:], (nfc[istation],nlead_time))
        data_tmp['v10m'][:,:] = np.reshape(nc_fid['v10m'][:,:,ilead:], (nfc[istation],nlead_time))
        data_tmp['td2m'][:,:] = np.reshape(nc_fid['td2m'][:,:,ilead:], (nfc[istation],nlead_time))
        data_tmp['mslp'][:,:] = np.reshape(nc_fid['mslp'][:,:,ilead:], (nfc[istation],nlead_time))
        data_tmp['pwat'][:,:] = np.reshape(nc_fid['pwat'][:,:,ilead:], (nfc[istation],nlead_time))
        data_tmp['gh500'][:,:] = np.reshape(nc_fid['gh500'][:,:,ilead:], (nfc[istation],nlead_time))
        data_tmp['gh700'][:,:] = np.reshape(nc_fid['gh700'][:,:,ilead:], (nfc[istation],nlead_time))
        data_tmp['gh850'][:,:] = np.reshape(nc_fid['gh850'][:,:,ilead:], (nfc[istation],nlead_time))
        data_tmp['t500'][:,:] = np.reshape(nc_fid['t500'][:,:,ilead:], (nfc[istation],nlead_time))
        data_tmp['t700'][:,:] = np.reshape(nc_fid['t700'][:,:,ilead:], (nfc[istation],nlead_time))
        data_tmp['t850'][:,:] = np.reshape(nc_fid['t850'][:,:,ilead:], (nfc[istation],nlead_time))
        data_tmp['u500'][:,:] = np.reshape(nc_fid['u500'][:,:,ilead:], (nfc[istation],nlead_time))
        data_tmp['u700'][:,:] = np.reshape(nc_fid['u700'][:,:,ilead:], (nfc[istation],nlead_time))
        data_tmp['u850'][:,:] = np.reshape(nc_fid['u850'][:,:,ilead:], (nfc[istation],nlead_time))
        data_tmp['v500'][:,:] = np.reshape(nc_fid['v500'][:,:,ilead:], (nfc[istation],nlead_time))
        data_tmp['v700'][:,:] = np.reshape(nc_fid['v700'][:,:,ilead:], (nfc[istation],nlead_time))
        data_tmp['v850'][:,:] = np.reshape(nc_fid['v850'][:,:,ilead:], (nfc[istation],nlead_time))
        data_tmp['w700'][:,:] = np.reshape(nc_fid['w700'][:,:,ilead:], (nfc[istation],nlead_time))


        # Calculate consistent averages for all lead times
        # The averages are saved for alternating 3h to 6h averages
        # Lead time +3h is for 0-3h, lead time +6h is for 0-6h
        # x03h = average for +0h to +3h
        # x06h = average for +0h to +6h
        # x36h = average for +3h to +6h
        # x36h = 2*x06h - x03h
        # It is lead times +6h, +12h, +18h, etc. that are 6h averages and
        # are the ones that need to be recalculated

        # Get radiation
        dswrf = np.reshape(nc_fid['dswrf'][:,:,ilead:], (nfc[istation],nlead_time))
        dlwrf = np.reshape(nc_fid['dlwrf'][:,:,ilead:], (nfc[istation],nlead_time))
        uswrf = np.reshape(nc_fid['uswrf'][:,:,ilead:], (nfc[istation],nlead_time))
        ulwrf = np.reshape(nc_fid['ulwrf'][:,:,ilead:], (nfc[istation],nlead_time))
        # Re-calculate consistent averages
        dswrf[:,1::2] = 2*dswrf[:,1::2] - dswrf[:,0::2]
        dlwrf[:,1::2] = 2*dlwrf[:,1::2] - dlwrf[:,0::2]
        uswrf[:,1::2] = 2*uswrf[:,1::2] - uswrf[:,0::2]
        ulwrf[:,1::2] = 2*ulwrf[:,1::2] - ulwrf[:,0::2]
        # Calculate netto long and short wave radiation: >0 = up; <0 = down
        data_tmp['nlwrf'][:,:] = ulwrf - dlwrf
        data_tmp['nswrf'][:,:] = uswrf - dswrf


        # Get heat fluxes
        lhtfl = np.reshape(nc_fid['lhtfl'][:,:,ilead:], (nfc[istation],nlead_time))
        shtfl = np.reshape(nc_fid['shtfl'][:,:,ilead:], (nfc[istation],nlead_time))
        # Re-calculate consistent averages
        lhtfl[:,1::2] = 2*lhtfl[:,1::2] - lhtfl[:,0::2]
        shtfl[:,1::2] = 2*shtfl[:,1::2] - shtfl[:,0::2]
        data_tmp['lhtfl'][:,:] = lhtfl
        data_tmp['shtfl'][:,:] = shtfl


        # Get cloud cover
        tcc = np.reshape(nc_fid['tcc'][:,:,ilead:], (nfc[istation],nlead_time))
        # Re-calculate consistent averages
        tcc[:,1::2] = 2*tcc[:,1::2] - tcc[:,0::2]
        # Need to make sure that tcc is alway minimum 0 and maximum 100
        tcc[tcc > 100.] = 100.;        tcc[tcc < 0.] = 0.
        data_tmp['tcc'][:,:] = tcc


        # Calculate wind speed
        data_tmp['wspd10m'] = np.sqrt(data_tmp['u10m']**2 + data_tmp['v10m']**2)

        # Calculate 10m wind direction variable from u10m and v10m
        data_tmp['wsdir10m'] = calculate_wsdir(data_tmp['u10m'],data_tmp['v10m'])

        # Extract data from NetCDF file
        issue_time = np.reshape(nc_fid['issue_time'][:,:,ilead:].astype(np.float64), (nfc[istation],nlead_time))
        valid_time = np.reshape(nc_fid['valid_time'][:,:,ilead:].astype(np.float64), (nfc[istation],nlead_time))
        data_tmp['issue_unix'][:,:] = issue_time
        data_tmp['valid_unix'][:,:] = valid_time


        #================================
        # Total accumulated precipitation
        #================================
        tp = np.reshape(nc_fid['tp'][:,:,ilead:], (nfc[istation],nlead_time))
        tp[:,1:] = tp[:,1:] - tp[:,0:-1]
        data_tmp['tp'] = tp


        # Turns out that I actually need to quality control the GFS data
        # Found bad MSLP value (+others) in one of the lead times from the forecast from 2020-01-11 18 UTC
        data_tmp = nwp_quality_check(data_tmp,gfs_names)

        # Add imei number
        data_tmp['imei'][:,:] = nc_fid['station_id']

        # Close the netCDF file
        nc_fid.close()


        # The ML model does not like missing values/nans
        # I know for a fact that one lead time in one of the forecasts issue 2020-01-31
        # and one of the forecasts issued 2019-10-10 have a missing lead time file,
        # which is why I created missing values for any missing lead time files
        # This means that I need to interpolate to get a value for these lead times!
#        print('Number of missing data BEFORE:', np.sum(np.isnan(data_tmp['t2m'])))
        for name in gfs_names:
            data_tmp = nwp_missing_data_interpolation(data_tmp,name)
            if np.sum(np.isnan(data_tmp[name])) > 0:
                print(name, ': still have missing data - EXIT')
                sys.exit()
#        print('Number of missing data AFTER:', np.sum(np.isnan(data_tmp['t2m'])))

        # Append to the data list
        data.append(data_tmp)


    return data



def ReadStationOBS(nstation,var_names,var_types,all_files):
    """
    Function for reading in observation data at station location from HDF file format
    """

    # Create structured data type
    array_type_list = []
    for i in range(len(var_names)):
        array_type_list.append((var_names[i],var_types[i]))

    # Create list for storing the data since the locations all have different number of observations
    data = []

    # Loop over number of stations
    for istation in range(nstation):
        print('Working on station number ' + str(istation))
        # Read data from netCDF files
        nc_f = all_files[istation]  # Your filename
        nc_fid = h5py.File(nc_f, 'r')
        obs_len = nc_fid['t2m'].shape[0]

        # Create structured arrays
        data_tmp = np.zeros((obs_len),dtype=array_type_list)

        # Variables
        data_tmp['t2m'] = nc_fid['t2m'][:] + celsius2kelvin
        data_tmp['wind_speed'] = nc_fid['wind_avg'][:]
        data_tmp['pres'] = nc_fid['pres'][:]
        data_tmp['precp'] = nc_fid['precp'][:]

        # Extract data from NetCDF file
        time = nc_fid['time'][:]

        # Need to process the time and rearrange/reshape the data
        data_tmp['unix'] = time # seconds since 1970, 1 Jan
#        data_tmp['time'] = [dt.datetime.utcfromtimestamp(time[itime]).strftime('%Y%m%d%H') for itime in range(obs_len)]
        data_tmp['datetime'] = [dt.datetime.strptime(str(dt.datetime.utcfromtimestamp(time[itime]).strftime('%Y%m%d%H%M%S')),'%Y%m%d%H%M%S') \
                                for itime in range(obs_len)]

        # Add IMEI number
        data_tmp['imei'][:] = nc_fid['station_id']
        data_tmp['lat'][:] = nc_fid['lat']#[:]
        data_tmp['lon'][:] = nc_fid['lon']#[:]

        # Close the netCDF file
        nc_fid.close()

        # Need to match to GFS output frequency
        temporal_interval = 3*60 # 3 hours (in minutes)
        data_tmp = temporal_match(data_tmp,temporal_interval,verbose=False)


        # Remove datetime and time from the dataframe
        for istation in range(nstation):
            data_tmp = remove_structured_field_name(data_tmp,'datetime')
#            data_tmp = remove_structured_field_name(data_tmp,'time')


        #================================================================================
        # NOTE: This part with missing data should be removed when the new netCDF files
        #       have been constructed, since they implement the interpolation on a 10 min
        #       basis. Here we get the interpolation on a 3-hourly basis, which is not
        #       optimal. However, Ill go with it for now in order to proceed with work
        #================================================================================
        # Put all the missing t2m values to nan
        stat = 't2m'
        if np.sum(data_tmp[stat] <= -9000) > 0:
            print('missing data --> exit!')
            sys.exit()
#        print('Number of missing t2m data=', np.sum(data_tmp[stat] <= -9000))


        # Append data to list
        data.append(data_tmp)


    return data



def temporal_match(data,temporal_interval,verbose):
    """
    Match the observation data to specified temporal output

    temporal_interval = temporal interval to which to match the osbervational record (in minute intervals)
    """
    if verbose:
        print('Match observations to chosen temporal output frequency')

    # Convert from numpy array to pandas datafram
    data_tmp = pd.DataFrame(data)

    # Want to only output the data in the same frequency "temporal_interval"
    # 1) Create time array with datetimes with temporal_interval output
    #    A) Get start date rounded up to nearest temporal_interval
    start_date = pd.Timestamp(data_tmp['datetime'][0]).to_pydatetime()
    year, month, day, hour, minute, second = start_date.year, start_date.month, start_date.day, start_date.hour, start_date.minute, start_date.second
    start_date = dt.datetime(year,month,day,hour,minute,second)
    delta = dt.timedelta(minutes=temporal_interval)
    start_date = start_date + (dt.datetime.min - start_date) % delta
    #    B) Get end date rounded down to nearest temporal_interval
    N = data_tmp.shape[0]
    end_date = pd.Timestamp(data_tmp['datetime'][data_tmp.shape[0]-1]).to_pydatetime()
    year, month, day, hour, minute, second = end_date.year, end_date.month, end_date.day, end_date.hour, end_date.minute, end_date.second
    end_date = dt.datetime(year,month,day,hour,minute,second)
    delta = dt.timedelta(minutes=temporal_interval)
    end_date = end_date - (end_date - dt.datetime.min) % delta
    #    C) Get number of temporal_interval elements in list
    diff_temporal = int( (end_date - start_date).total_seconds() / 60 / temporal_interval )
    datetime_temporal = [start_date + dt.timedelta(minutes=temporal_interval*x) for x in range(diff_temporal)]


    # Create new dataframe
    data_cols = data_tmp.columns.to_list()
    data_types = data_tmp.dtypes.to_list()
    idx = np.argwhere(np.array(data_types) == 'datetime64[ns]')[:,0]
    if len(idx) > 0:
        data_types[idx[0]] = 'datetime64[s]'
    obs_cols = data_cols + ['observation_datetime']
    # Drop unix and time stamp
#    obs_cols.remove('unix')
#    obs_cols.remove('timestamp')
    obs_data = pd.DataFrame(columns=obs_cols)

    # Add the columns - except the temporal ones
    for col in data_cols:
        if ( (col == 'timestamp') | (col == 'datetime') | (col == 'unix')):
            obs_data['observation_' + col] = data_tmp[col]
        else:
            obs_data[col] = data_tmp[col]

    # Add datetime, which will be used for indexing
    obs_data['datetime'] = data_tmp['datetime'].dt.round('10min')

    # Get the duplicate values
    dup = obs_data.duplicated(subset=['datetime']).values

    # Remove the duplicates
    obs_data = obs_data.loc[~dup,:].sort_values('datetime')

    # Set index to datetime
    obs_data = obs_data.set_index(['datetime'])

    # Set new index based on the new temporal resolution - will fill nan for the missing entries!
    new_index = datetime_temporal
    obs_data = obs_data.reindex(new_index)

    # Get the new index as a column
    obs_data = obs_data.reset_index()

    # Add timestamp and unix
    obs_data['unix'] = (obs_data['datetime'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    obs_data['timestamp'] = [unix2date(unix,fmt_data) for unix in obs_data['unix'].values.tolist()]


    # Convert from dataframe to structured array
    array_type_list = []
    for i in range(len(data_cols)):
        array_type_list.append((data_cols[i],data_types[i]))

    # Create structured array
    data_new = np.zeros((obs_data.shape[0]),dtype=array_type_list)

    for col in data_cols:
        data_new[col] = obs_data[col].values


    return data_new



def nwp_missing_data_interpolation(data_tmp,field_name):
    """
    Function for handling missing data in the GFS data record.

    NOTE: nan_helper only works for 1d arrays!
    """
    # Logical for saving data
    nc_file_save = True
    process_data = True

    # Dimensions
    nfc = data_tmp[field_name].shape[0]
    nlead_time = data_tmp[field_name].shape[1]

    # Loop over forecasts and for those forecasts for which we have nan values, interpolate!
    for ifc in range(nfc):
        if np.sum(np.isnan(data_tmp[field_name][ifc,:])) == nlead_time:
            # Don't interpolate since all of the data is missing for this forecast
            pass
        elif np.sum(np.isnan(data_tmp[field_name][ifc,:])) > 0:
            # Get the nans (logicals) and the nan index function
            nans, nan_index_func = nan_helper(data_tmp[field_name][ifc,:])

            # Get indices for nan values
            nan_idx = nan_index_func(nans)

            # Get ranges with consecutive nan values [start, end)
            ranges = consecutive_number_ranges(nan_idx)

            # Calculate the length of these ranges (remember that it's an open end interval!)
            ndata_nan_ranges = [range_vals[1] - range_vals[0] for range_vals in ranges]

            data_tmp[field_name][ifc,nans]= np.interp(nan_index_func(nans), nan_index_func(~nans), data_tmp[field_name][ifc,~nans])

    return data_tmp



def nwp_quality_check(data,gfs_names):
    """
    Quality control of the NWP data.
    Will only be checking plausible values, to get the obviously erroneous data,
    such at a MSLP of 0 hPa.
    """
    # Dimensions
    nfc = data['t2m'].shape[0]
    nlead_time = data['t2m'].shape[1]

    # Limits
    T2m_min = -40. + 273.15     # Kelvin
    T2m_max =  40. + 273.15     # Kelvin
    Tp_min = -60. + 273.15      # Kelvin
    Tp_max =  30. + 273.15      # Kelvin
    WS10m_min = 0.              # m/s
    WS10m_max = 40.             # m/s
    W10m_min = -40.             # m/s
    W10m_max = 40.              # m/s
    Wp_min = -100.              # m/s
    Wp_max = 100.               # m/s
    WSdir_min = 0.              # degrees
    WSdir_max =  360.           # degrees
    w_min = -100.               # Pa/s
    w_max = 100.                # Pa/s
    RH2m_min = 0.               # %
    RH2m_max = 101.             # %
    Q2m_min = 0.                # g/kg
    Q2m_max = 40.               # g/kg
    Td2m_min = -40. + 273.15    # Kelvin
    Td2m_max = 40. + 273.15     # Kelvin
    pres_min = 92000.           # Pa
    pres_max = 107000.          # Pa
    cc_min = 0.                 # %
    cc_max = 100.               # %
    precp_min = 0.              # mm

    # Plausibility check
    # Since the variables are tied together through the atmospheric equations,
    # I will  mask all when one if not plausible
    mask = ( ( ( (data['t2m']      - T2m_min  ) < 0. ) | ( (T2m_max   - data['t2m']     ) < 0. ) ) |
             ( ( (data['t500']     - Tp_min   ) < 0. ) | ( (Tp_max    - data['t500']    ) < 0. ) ) |
             ( ( (data['t700']     - Tp_min   ) < 0. ) | ( (Tp_max    - data['t700']    ) < 0. ) ) |
             ( ( (data['t850']     - Tp_min   ) < 0. ) | ( (Tp_max    - data['t850']    ) < 0. ) ) |
             ( ( (data['u10m']     - W10m_min ) < 0. ) | ( (W10m_max  - data['u10m']    ) < 0. ) ) |
             ( ( (data['u500']     - Wp_min   ) < 0. ) | ( (Wp_max    - data['u500']    ) < 0. ) ) |
             ( ( (data['u700']     - Wp_min   ) < 0. ) | ( (Wp_max    - data['u700']    ) < 0. ) ) |
             ( ( (data['u850']     - Wp_min   ) < 0. ) | ( (Wp_max    - data['u850']    ) < 0. ) ) |
             ( ( (data['v10m']     - W10m_min ) < 0. ) | ( (W10m_max  - data['v10m']    ) < 0. ) ) |
             ( ( (data['v500']     - Wp_min   ) < 0. ) | ( (Wp_max    - data['v500']    ) < 0. ) ) |
             ( ( (data['v700']     - Wp_min   ) < 0. ) | ( (Wp_max    - data['v700']    ) < 0. ) ) |
             ( ( (data['v850']     - Wp_min   ) < 0. ) | ( (Wp_max    - data['v850']    ) < 0. ) ) |
             ( ( (data['wspd10m']  - WS10m_min) < 0. ) | ( (WS10m_max - data['wspd10m'] ) < 0. ) ) |
             ( ( (data['wsdir10m'] - WSdir_min) < 0. ) | ( (WSdir_max - data['wsdir10m']) < 0. ) ) |
             ( ( (data['w700']     - w_min    ) < 0. ) | ( (w_max     - data['w700']    ) < 0. ) ) |
             ( ( (data['rh2m']     - RH2m_min ) < 0. ) | ( (RH2m_max  - data['rh2m']    ) < 0. ) ) |
             ( ( (data['q2m']      - Q2m_min  ) < 0. ) | ( (Q2m_max   - data['q2m']     ) < 0. ) ) |
             ( ( (data['td2m']     - Td2m_min ) < 0. ) | ( (Td2m_max  - data['td2m']    ) < 0. ) ) |
             ( ( (data['tcc']      - cc_min   ) < 0. ) | ( (cc_max    - data['tcc']     ) < 0. ) ) |
             ( ( (data['mslp']     - pres_min ) < 0. ) | ( (pres_max  - data['mslp']    ) < 0. ) ) |
             (   (data['pwat']     - precp_min) < 0. ) )

    for name in gfs_names:
        data[name][mask] = np.nan


    return data



def read_training_datasets(ifile,read_dict={'data_train':True,'data_train_1d':True,'extra_data_train':True, \
                                            'data_raw_train':True,'extra_data_raw_train':True, \
                                            'station_train_index_shifted_1d':True}):
    """
    Function for reading training data saved on format hdf5
    """
    # Open file
    f = h5py.File(ifile,'r')

    for group in f.keys() :
        if read_dict[group]:
            print('Extracting data from group: ', group)
            if group == 'data_train_1d':
                data = f[group]['data_train_1d'][:]
            elif group == 'station_train_index_shifted_1d':
                data = f[group]['station_train_index_shifted_1d'][:]
            else:
                data = []
                members = list(f[group].keys())
                members.sort(key=float)
                for member in members:
                    data.append(f[group][member][:]) # adding [:] returns a numpy array
        else:
            data = []

        # Assign to the correct list
        if group == 'data_train_1d':
            data_train_1d = data
        elif group == 'data_train':
            data_train = data
        elif group == 'extra_data_train':
            extra_data_train = data
        elif group == 'data_raw_train':
            data_raw_train = data
        elif group == 'extra_data_raw_train':
            extra_data_raw_train = data
        elif group == 'station_train_index_shifted_1d':
            station_train_index_shifted_1d = data


    return data_train, data_train_1d, extra_data_train, data_raw_train, extra_data_raw_train, station_train_index_shifted_1d



def read_val_datasets(ifile,read_dict={'data_val':True,'data_val_1d':True,'extra_data_val':True, \
                                        'data_raw_val':True,'extra_data_raw_val':True, \
                                        'station_val_index_shifted_1d':True}):
    """
    Function for reading validation data saved on format hdf5
    """
    # Open file
    f = h5py.File(ifile,'r')

    for group in f.keys() :
        if read_dict[group]:
            print('Extracting data from group: ', group)
            if group == 'data_val_1d':
                data = f[group]['data_val_1d'][:]
            elif group == 'station_val_index_shifted_1d':
                data = f[group]['station_val_index_shifted_1d'][:]
            else:
                data = []
                members = list(f[group].keys())
                members.sort(key=float)
                for member in members:
                    data.append(f[group][member][:]) # adding [:] returns a numpy array
        else:
            data = []

        # Assign to the correct list
        if group == 'data_val_1d':
            data_val_1d = data
        elif group == 'data_val':
            data_val = data
        elif group == 'extra_data_val':
            extra_data_val = data
        elif group == 'data_raw_val':
            data_raw_val = data
        elif group == 'extra_data_raw_val':
            extra_data_raw_val = data
        elif group == 'station_val_index_shifted_1d':
            station_val_index_shifted_1d = data


    return data_val, data_val_1d, extra_data_val, data_raw_val, extra_data_raw_val, station_val_index_shifted_1d



def read_test_datasets(ifile,read_dict={'data_test':True,'data_test_1d':True,'extra_data_test':True, \
                                              'data_raw_test':True,'extra_data_raw_test':True}):
    """
    Function for reading test data saved on format hdf5
    """
    # Open file
    f = h5py.File(ifile,'r')

    for group in f.keys() :
        if read_dict[group]:
            print('Extracting data from group: ', group)
            if group == 'data_test_1d':
                data = f[group]['data_test_1d'][:]
            else:
                data = []
                members = list(f[group].keys())
                members.sort(key=float)
                for member in members:
                    data.append(f[group][member][:]) # adding [:] returns a numpy array
        else:
            data = []

        # Assign to the correct list
        if group == 'data_test_1d':
            data_test_1d = data
        elif group == 'data_test':
            data_test = data
        elif group == 'extra_data_test':
            extra_data_test = data
        elif group == 'data_raw_test':
            data_raw_test = data
        elif group == 'extra_data_raw_test':
            extra_data_raw_test = data

    return data_test, data_test_1d, extra_data_test, data_raw_test, extra_data_raw_test



def read_auxiliary_data(ifile):
    """
    Function for reading auxiliary data for train, test and validation datasets
    """

    # Open pickle file
    f = open(ifile,'rb')
    save_tuple = pickle.load(f)
    f.close()
    mu, std, station_train, station_test, station_val = save_tuple

    return mu, std, station_train, station_test, station_val



def read_NWP_OBS_hdf5(ifile):
    """
    Function for reading GFS and OBS data from hdf5 file
    """
    f = h5py.File(ifile,'r')

    data = []
    for group in f.keys() :
        print('Extracting data from group: ', group)
        members = list(f[group].keys())
        members.sort(key=float)
        for member in members:
            data.append(f[group][member][:]) # adding [:] returns a numpy array

    # Close the file
    f.close()


    return data



def save_NWP_OBS_hdf5(ifile,group_name,data):
    """
    Function for saving GFS and OBS data on hdf5 format with datasets under group_name
    """
    # Dimensions
    nstation = len(data)

    # Open file
    f = h5py.File(ifile,'w')

    # Create dictionary for saving the data
    data_dict = { str(istation) : data[istation] for istation in range(nstation) }

    # Create HDF group
    grp=f.create_group(group_name)

    # Add members (locations/stations)
    for k,v in data_dict.items():
        grp.create_dataset(k,data=v)

    # Close the file
    f.close()

    # Delete unnecessary data
    del data_dict, k, v



def save_training_datasets(ofile,data_list,data_groups,data_1d,station_index_1d):
    """
    Function for saving training data on hdf5 format with datasets under group_name
    """
    # Dimensions
    nstation = len(data_list[0])

    # Save the training data in hdf5 format
    f = h5py.File(ofile,'w')

    for idata,data in enumerate(data_list):
        print('   ---Saving ', data_groups[idata])
        # Create dictionary for saving the data
        data_dict = { str(istation) : data[istation] for istation in range(nstation) }

        # Create HDF group
        grp=f.create_group(data_groups[idata])

        # Add members (locations/stations)
        for k,v in data_dict.items():
            grp.create_dataset(k,data=v)

        # Delete unnecessary data
        del data_dict

    # Delete already-used data
    del data_list, data_groups, data

    print('   ---Saving data_train_1d')
    # Create HDF group
    grp=f.create_group('data_train_1d')

    # Add data
    grp.create_dataset('data_train_1d',data=data_1d)

    # Delete already-used data
    del data_1d

    print('   ---Saving station_train_index_shifted_1d')
    # Create HDF group
    grp=f.create_group('station_train_index_shifted_1d')

    # Add data
    grp.create_dataset('station_train_index_shifted_1d',data=station_index_1d)

    # Delete already-used data
    del station_index_1d

    # Close the file
    f.close()



def save_val_datasets(ofile,data_list,data_groups,data_1d,station_index_1d):
    """
    Function for saving val data on hdf5 format with datasets under group_name
    """
    # Dimensions
    nstation = len(data_list[0])

    # Save the val data in hdf5 format
    f = h5py.File(ofile,'w')

    for idata,data in enumerate(data_list):
        print('   ---Saving ', data_groups[idata])
        # Create dictionary for saving the data
        data_dict = { str(istation) : data[istation] for istation in range(nstation) }

        # Create HDF group
        grp=f.create_group(data_groups[idata])

        # Add members (locations/stations)
        for k,v in data_dict.items():
            grp.create_dataset(k,data=v)

        # Delete unnecessary data
        del data_dict

    # Delete already-used data
    del data_list, data_groups, data

    print('   ---Saving data_val_1d')
    # Create HDF group
    grp=f.create_group('data_val_1d')

    # Add data
    grp.create_dataset('data_val_1d',data=data_1d)

    # Delete already-used data
    del data_1d

    print('   ---Saving station_val_index_shifted_1d')
    # Create HDF group
    grp=f.create_group('station_val_index_shifted_1d')

    # Add data
    grp.create_dataset('station_val_index_shifted_1d',data=station_index_1d)

    # Delete already-used data
    del station_index_1d

    # Close the file
    f.close()



def save_test_datasets(ofile,data_list,data_groups,data_1d):
    """
    Function for saving test data on hdf5 format with datasets under group_name
    """
    # Dimensions
    nstation = len(data_list[0])

    # Save the test data in hdf5 format
    f = h5py.File(ofile,'w')

    for idata,data in enumerate(data_list):
        print('   ---Saving ', data_groups[idata])
        # Create dictionary for saving the data
        data_dict = { str(istation) : data[istation] for istation in range(nstation) }

        # Create HDF group
        grp=f.create_group(data_groups[idata])

        # Add members (locations/stations)
        for k,v in data_dict.items():
            grp.create_dataset(k,data=v)

        # Delete unnecessary data
        del data_dict

    # Delete already-used data
    del data_list, data_groups, data

    print('   ---Saving data_test_1d')
    # Create HDF group
    grp=f.create_group('data_test_1d')

    # Add data
    grp.create_dataset('data_test_1d',data=data_1d)

    # Delete already-used data
    del data_1d

    # Close the file
    f.close()



def save_auxiliary_data(ofile,save_tuple):
    """
    Function for saving auxiliary/generic data
    """
    # Open for pickling
    f = open(ofile,'wb')
    pickle.dump(save_tuple,f)
    f.close()
