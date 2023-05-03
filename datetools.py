#=========================
# Generic library imports
#=========================
import datetime
import julian
import sys
import time

#=========================
# Local imports
#=========================
#from parameters import *



def julian2datetime(jd):
    """
    Convert from julian to datetime

    Input:
    + jd: Julian time

    Output:
    + date_time: Datetime
    """
    return julian.from_jd(jd,fmt='jd')


def datetime2julian(date_time):
    """
    Convert from datetime to julian

    Input:
    + date_time: Datetime

    Output:
    + jd: Julian time
    """
    return julian.to_jd(date_time,fmt='jd')


def time2julian(timestr,fmt):
    """
    Convert from time (string) to julian

    Input:
    + timestr: Time (str)
    + fmt: Format of the time string

    Output:
    + jd: Julian time
    """
    return julian.to_jd(datetime.datetime.strptime(timestr,fmt),fmt='jd')


def julian2time(jd,fmt):
    """
    Convert from julian to time (string)

    Input:
    + jd: Julian time
    + fmt: Format of the time string

    Output:
    + timestr: Time (str)
    """
    return datetime.datetime.strftime(julian.from_jd(jd,fmt='jd'),fmt)


def time2datetime(timestr,fmt):
    """
    Convert from time (str) to datetime

    Input:
    + timestr: Time (str)
    + fmt: Format of time string

    Output:
    + date_time: Datetime
    """
    return datetime.datetime.strptime(timestr, fmt)


def datetime2datecomponents(date_time):
    """
    Get year, month, day, hour, minute, second from datetime

    Input:
    + date_time: Datetim

    Output:
    + year: Year
    + month: Month
    + day: Day
    + hour: Hour
    + minute: Minute
    + second: Second
    """
    year = date_time.year
    month = date_time.month
    day = date_time.day
    hour = date_time.hour
    minute = date_time.minute
    second = date_time.second

    return year, month, day, hour, minute, second
    

def date2unix(timestr,fmt):
    """
    Get unix time from time string

    Input:
    + timestr: Time string
    + fmt: Format of the time string

    Output:
    + unix_time: Unix time
    """
    date_time = datetime.datetime.strptime(timestr,fmt)
    unix = date_time.replace(tzinfo=datetime.timezone.utc).timestamp()

    return unix



def unix2date(unix,fmt):
    """
    Get time string from unix time

    Input:
    + unix_time: Unix time
    + fmt: Format of the time string

    Output:
    + timestr: Time string
    """

    # Convert to datetime
    date_time = datetime.datetime.utcfromtimestamp(unix)

    # Convert to time string
    timestr = datetime.datetime.strftime(date_time, fmt)

    return timestr



def unix2datetime(unix):
    """
    Get datetime from unix time

    Input:
    + unix_time: Unix time

    Output:
    + date_time: Datetime
    """

    # Convert to datetime
    date_time = datetime.datetime.utcfromtimestamp(unix)

    return date_time



def datetime2unix(date_time):
    """
    Get unix time from datetime

    Input:
    + date_time: Datetime

    Output:
    + unix_time: Unix time
    """

    # Convert to unix time
    unix_time = (date_time - datetime.datetime(1970,1,1)).total_seconds()

    return unix_time

