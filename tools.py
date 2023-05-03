"""
NAME:
         tools

PURPOSE:
         Utility tools and functions

CONTAINS:
         does_directory_exist
         create_dir
         convert_unix_times
         calculate_day_of_year
         calculate_hour_of_day

"""

#==========================
# Standard library imports
#==========================
import numpy as np
from os import mkdir
from os.path import isdir
import pandas as pd
import sys



def does_directory_exist(dir):
    """
    Check if directory exist

    Arguments:
    ----------
    + dir: directory to check

    Returns:
    --------
    isdir(dir): logical
    """
    return isdir(dir)



def create_dir(dir):
    """
    Create a new directory

    Arguments:
    ----------
    + dir: directory to check

    Returns:
    --------
    """
    mkdir(dir)



def convert_unix_times(unix_time):
    """
    Convert from unix time to time string and datetime

    Arguments:
    ----------
    + unix_time: Unix time

    Returns:
    --------
    + timestring_time: Time string array
    + datetime_time: Datetime array
    """
    # Get dimensions of unix_time
    dims = unix_time.shape#[0]

    # Convert to pandas data frame
    df = pd.DataFrame(data=unix_time.flatten(),columns=['date'])
    # Convert from unix time to datetime
    df['date'] = pd.to_datetime(df['date'],unit='s')
    # Convert to pandas datetime index
    dates = pd.DatetimeIndex(df['date'])

    # Get the datetime array
    datetime_time = np.array(dates.tolist()).reshape(dims)#.T
#    datetime_time = np.array([np.datetime64(date) for date in dates.tolist()]).reshape(dims)
    # Get the dates and convert them to strings
    df = pd.DataFrame(data=np.char.mod('%04d', dates.year.values),columns=['dates'])
    df['dates'] = df['dates'].astype(str)
    years = df.to_numpy()
    df = pd.DataFrame(data=np.char.mod('%02d', dates.month.values),columns=['dates'])
    df['dates'] = df['dates'].astype(str)
    months = df.to_numpy()
    df = pd.DataFrame(data=np.char.mod('%02d', dates.day.values),columns=['dates'])
    df['dates'] = df['dates'].astype(str)
    days = df.to_numpy()
    df = pd.DataFrame(data=np.char.mod('%02d', dates.hour.values),columns=['dates'])
    df['dates'] = df['dates'].astype(str)
    hours = df.to_numpy()

    # Get the time stamp
    timestamps = years + months + days + hours
    timestring_time = timestamps.reshape(dims).T

    return timestring_time, datetime_time



def calculate_day_of_year(unix_time):
    """
    Calculate day of year

    Arguments:
    ----------
    + unix_time: Unix timestamp


    Returns:
    --------
    doy: Day of year array
    """
    # Dimensions
    dims = unix_time.shape
    # Convert to pandas data frame
    df = pd.DataFrame(data=unix_time.flatten(),columns=['date'])
    # Convert from unix time to datetime
    df['date'] = pd.to_datetime(df['date'],unit='s')
    # Convert to pandas datetime index
    doy = np.array(pd.DatetimeIndex(df['date']).dayofyear.tolist()).reshape(dims)

    return doy



def calculate_hour_of_day(unix_time):
    """
    Calculate hour of day

    Arguments:
    ----------
    + unix_time: Unix timestamp


    Returns:
    --------
    hod: Hour of day array
    """
    # Dimensions
    dims = unix_time.shape
    # Convert to pandas data frame
    df = pd.DataFrame(data=unix_time.flatten(),columns=['date'])
    # Convert from unix time to datetime
    df['date'] = pd.to_datetime(df['date'],unit='s')
    # Convert to pandas datetime index
    hod = np.array(pd.DatetimeIndex(df['date']).hour.tolist()).reshape(dims)

    return hod



def nan_helper(y):
    """
    Helper to handle indices and logical indices of NaNs.

    Arguments:
    ----------
        + y, 1d numpy array with possible NaNs

    Returns:
    --------
        + nans, logical indices of NaNs
        + index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]



def consecutive_number_ranges(nums):
    """
    Calculate ranges with consecutive numbers

    Arguments:
    ----------
    + nums: List of numbers

    Returns:
    --------
    + List with start and end index (open end) for ranges with consecutive numbers
    """
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])

    # Return ranges - with open end; [start, end)
    return [(s, e+1) for s, e in zip(edges, edges)]
#    # Return ranges - with closed end; [start, end]
#    return list(zip(edges, edges))



def remove_structured_field_name(a, name):
    """
    Remove field from structred array

    Arguments:
    ----------
    + a: Structured array
    + name: Name of field to remove

    Returns:
    + b: Structured array, equal to a, but without field "name"
    """
    names = list(a.dtype.names)
    if name in names:
        names.remove(name)
    b = a[names]

    return b



def shift(arr, num, fill_value=np.nan):
    """
    Shift 1D array

    Arguments:
    + arr: Input array, 1D
    + num: The number used for shifting the array
    + fill_value: Number to fill the shifted spot(s) without data with

    Returns:
    arr: Shifted array, 1D
    """
    arr = np.roll(arr,num)
    if num < 0:
        arr[num:] = fill_value
    elif num > 0:
        arr[:num] = fill_value
    return arr



def shift2d(arr, num, ax, fill_value=np.nan):
    """
    Shift 2D array

    Arguments:
    + arr: Input array, 1D
    + num: The number used for shifting the array
    + ax: Axis to shift on
    + fill_value: Number to fill the shifted spot(s) without data with

    Returns:
    arr: Shifted array, 2D
    """

    arr = np.roll(arr,num,ax)
    if ax == 0:
        if num < 0:
            arr[num:,:] = fill_value
        elif num > 0:
            arr[:num,:] = fill_value
    if ax == 1:
        if num < 0:
            arr[:,num:] = fill_value
        elif num > 0:
            arr[:,:num] = fill_value
    return arr



def roll_odd_data(dims,data_unrolled,fill_val):
    """
    Shift only the odd elements of the first axis in a 2D array

    Arguments:
    ----------
    + dims: Dimensions of the full array
    + data_unrolled: 2D array for which odd elements are to be shifted 
    + fill_value: Fill value


    Returns:
    + data_roll: 2D array for which odd elements have been shifted
    """
    # Initialize data
    data_roll = np.full((dims), fill_value=np.nan)
    # Even array
    data_even = data_unrolled[::2,:]
    # Odd array
    data_odd = data_unrolled[1::2,:]
    # Roll the odd array
    data_odd_roll = shift2d(data_odd,1,1,fill_value=fill_val)#np.expand_dims(data_odd[:,0],1))
    # Add even and rolled odd array to new array
    data_roll[::2,:] = data_even
    data_roll[1::2,:] = data_odd_roll

    return data_roll


def cyclic_encoding(data,data_max):
    """
    Cyclic encoding of data

    Arguments:
    ----------
    + data: data to be encoded as cyclic
    + data_max: Max value for data (theorethical!)

    Returns:
    --------
    sin_data: Cyclic sine encoding of data
    cos_data: Cyclis cosine ecoding of data
    """

    sin_data = np.sin(2 * np.pi * data/data_max)
    cos_data = np.cos(2 * np.pi * data/data_max)


    return sin_data, cos_data

