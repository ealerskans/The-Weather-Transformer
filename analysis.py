"""
NAME:
         analysis

PURPOSE:
         Definitions for analysis functions 

CONTAINS:

"""
#===========================
# Standard library imports
#===========================
import datetime
from matplotlib import colors as mcolors
from matplotlib.animation import FuncAnimation
from matplotlib.cm import get_cmap
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.basemap import Basemap
import numpy as np
import pandas as pd
import sys
from tabulate import tabulate


#===============
# Local imports
#===============
from general import GetDataVariables



#===================
# Global variables
#===================
# Get names for structured array
stat_names, stat_types = GetDataVariables('stat')
# Define statistics structured arrays
stat_array_type_list = []
for i in range(len(stat_names)):
    stat_array_type_list.append((stat_names[i], stat_types[i]))

fc_output_interval = 3
fc_update = 6
nfc_cycles = int(24/fc_update)

# Seasons
seasons = [ [12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
nseasons = len(seasons)
season_name = [ 'DJF', 'MAM', 'JJA', 'SON']
season_labels = ['Winter (DJF)', 'Spring (MAM)', 'Summer (JJA)', 'Autumn (SON)']
    
# Months
months = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
nmonths = len(months)
month_labels = [ 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']



def centered_RMSE(YFC,YOBS):
    """
    Calculate the centered RMSE

    RMSE^2 = 1/N sum_{i=1}^{N} (YFC_i - YOBS_i)^2 = CRMSE^2 + BIAS^2

    CRMSE = sqrt( 1/N sum_{i=1}^{N} [ (YFC_i - mu_YFC) - (YOBS_i - mu_YOBS) ]^2 )

    BIAS = mu_YFC - mu_YOBS
    """

    YFCbias = np.nanmean(YFC,axis=0)
    YOBSbias = np.nanmean(YOBS,axis=0)

    CRMSE = np.sqrt(np.nanmean( ( (YFC - YFCbias) - (YOBS - YOBSbias) )**2,axis=0))

    return CRMSE



def statistics_measures(YFC,YOBS,stat):
    """
    Get statistics:
    + Bias (Mean Error, ME)
    + Absolute bias (Mean Absolute Error, MAE)
    + Standard deviation of bias
    + Standard deviation of absolute bias
    + RMSE (Root Mean Square Error)
    + r (Pearson's correlation coefficient)
    """

    YFCbias = np.nanmean(YFC,axis=0)
    YOBSbias = np.nanmean(YOBS,axis=0)

    stat['ME']      = np.nanmean(YFC-YOBS,axis=0)
    stat['MAE']     = np.nanmean(np.abs(YFC-YOBS),axis=0)
    stat['STD_ME']  = np.nanstd(YFC-YOBS,axis=0)
    stat['STD_MAE'] = np.nanstd(np.abs(YFC-YOBS),axis=0)
    stat['RMSE']    = np.sqrt(np.nanmean((YFC-YOBS)**2,axis=0))
    stat['r']       = np.nansum((YOBS-YOBSbias)*(YFC-YFCbias)) \
                      /( np.sqrt(np.nansum((YOBS-YOBSbias)**2)) \
                         *np.sqrt(np.nansum((YFC-YFCbias  )**2)) )

    return stat


def basemap_subplot(ax,lat_min,lat_max,lon_min,lon_max,lat_ticks,lon_ticks):
    """
    Basemap function
    """
    # Orthographic map projection with perspective of satellite looking down at 50N, 100W.
    m = Basemap(projection='mill', llcrnrlon=lon_min, llcrnrlat=lat_min, urcrnrlon=lon_max, \
                                   urcrnrlat=lat_max, resolution='i', suppress_ticks=False, \
                                   fix_aspect=False, ax=ax)
    # Convert from degree to map projection
    lon_ticks_proj, _ = m(lon_ticks, np.zeros(len(lon_ticks)))
    _, lat_ticks_proj = m(np.zeros(len(lat_ticks)), lat_ticks)
    # manually add ticks
    ax.set_xticks(lon_ticks_proj)
    ax.set_yticks(lat_ticks_proj)
    ax.tick_params(axis='both',which='major')
    # add ticks to the opposite side as well
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    # remove the tick labels
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    # Fill continents
    m.fillcontinents(color='k', zorder=1)
    # Draw coastlines, country boundaries
    m.drawcoastlines(linewidth=0.50, color='k')
    m.drawcountries(linewidth=0.50, color='k')
    # Draw lat/lon grid lines
    m.drawmeridians(lon_ticks, color='k', linewidth=0., labels=[1,0,0,1], fontsize=14)
    m.drawparallels(lat_ticks, color='k', linewidth=0., labels=[1,0,0,1], fontsize=14)

    return m




def CalculateOverallStatistics(ML_update_time,valid_time,t2m_obs,t2m_nwp,t2m_pred):
    """
    Calculate overall different statistics measures. 
    """

    # Dimensions
    nstation = len(t2m_pred)

    # Define statistics arrays
    stat_pred = np.full((nstation), fill_value=np.nan, dtype=stat_array_type_list)
    stat_nwp  = np.full((nstation), fill_value=np.nan, dtype=stat_array_type_list)
    nobs_pred = np.full((nstation), fill_value=np.nan, dtype=int)
    nobs_nwp  = np.full((nstation), fill_value=np.nan, dtype=int)

    Nmin = 60

    for istation in range(nstation):
        step = int(fc_update/fc_output_interval)
        mask_nwp = ( (ML_update_time[istation][::step,:] < valid_time[istation][::step,:]) )
        mask_pred = ( (ML_update_time[istation][:,:] < valid_time[istation][:,:]) )

        t2m_nwp_lead = t2m_nwp[istation][::step,:][mask_nwp].flatten()
        t2m_nwp_obs_lead = t2m_obs[istation][::step,:][mask_nwp].flatten()
        t2m_pred_lead  = t2m_pred[istation][:,:][mask_pred].flatten()
        t2m_pred_obs_lead = t2m_obs[istation][:,:][mask_pred].flatten()

        nobs_pred[istation] = np.sum(~np.isnan(t2m_pred_lead-t2m_pred_obs_lead))
        nobs_nwp[istation] = np.sum(~np.isnan(t2m_nwp_lead-t2m_nwp_obs_lead))
        if ( (nobs_pred[istation] > Nmin) & (nobs_nwp[istation] > Nmin) ):
            stat_pred[:][istation] = statistics_measures(t2m_pred_lead,t2m_pred_obs_lead,stat_pred[:][istation])
            stat_nwp[:][istation] = statistics_measures(t2m_nwp_lead,t2m_nwp_obs_lead,stat_nwp[:][istation])

    return stat_nwp, stat_pred, nobs_nwp, nobs_pred



def CalculateSeasonalOverallStatistics(ML_update_time,valid_time,valid_datetime,t2m_obs,t2m_nwp,t2m_pred):
    """
    Calculate overall different statistics measures. 
    """

    # Dimensions
    nstation = len(t2m_pred)
    nlead_time = t2m_pred[0].shape[1]

    # Define statistics arrays
    stat_pred = np.full((nmonths,nstation), fill_value=np.nan, dtype=stat_array_type_list)
    stat_nwp  = np.full((nmonths,nstation), fill_value=np.nan, dtype=stat_array_type_list)
    nobs_pred = np.full((nmonths,nstation), fill_value=np.nan, dtype=int)
    nobs_nwp  = np.full((nmonths,nstation), fill_value=np.nan, dtype=int)

    Nmin = 50

    for istation in range(nstation):
        nfc = t2m_pred[istation].shape[0]
        for iseason in range(nseasons):
            step = int(fc_update/fc_output_interval)
            mask_pred = np.full((nfc,nlead_time), fill_value=False)
            mask_nwp = np.full((int(nfc/step),nlead_time), fill_value=False)

            for ilead in range(nlead_time):
                mask_nwp[:,ilead] = ( ( ML_update_time[istation][::step,ilead] < valid_time[istation][::step,ilead] ) & \
                                      ( ( pd.DatetimeIndex(valid_datetime[istation][::step,ilead]).month == seasons[iseason][0] ) | \
                                        ( pd.DatetimeIndex(valid_datetime[istation][::step,ilead]).month == seasons[iseason][1] ) | \
                                        ( pd.DatetimeIndex(valid_datetime[istation][::step,ilead]).month == seasons[iseason][2] ) ) )
                mask_pred[:,ilead] = ( ( ML_update_time[istation][:,ilead] < valid_time[istation][:,ilead] ) & \
                                       ( ( pd.DatetimeIndex(valid_datetime[istation][:,ilead]).month == seasons[iseason][0] ) | \
                                         ( pd.DatetimeIndex(valid_datetime[istation][:,ilead]).month == seasons[iseason][1] ) | \
                                         ( pd.DatetimeIndex(valid_datetime[istation][:,ilead]).month == seasons[iseason][2] ) ) )

            t2m_nwp_lead = t2m_nwp[istation][::step,:][mask_nwp].flatten()
            t2m_nwp_obs_lead = t2m_obs[istation][::step,:][mask_nwp].flatten()
            t2m_pred_lead  = t2m_pred[istation][:,:][mask_pred].flatten()
            t2m_pred_obs_lead = t2m_obs[istation][:,:][mask_pred].flatten()

            nobs_pred[iseason,istation] = np.sum(~np.isnan(t2m_pred_lead-t2m_pred_obs_lead))
            nobs_nwp[iseason,istation] = np.sum(~np.isnan(t2m_nwp_lead-t2m_nwp_obs_lead))
            if ( (nobs_pred[iseason,istation] > Nmin) & (nobs_nwp[iseason,istation] > Nmin) ):
                stat_pred[:][iseason,istation] = statistics_measures(t2m_pred_lead,t2m_pred_obs_lead,stat_pred[:][iseason,istation])
                stat_nwp[:][iseason,istation] = statistics_measures(t2m_nwp_lead,t2m_nwp_obs_lead,stat_nwp[:][iseason,istation])

    return stat_nwp, stat_pred, nobs_nwp, nobs_pred





def CalculateLeadTimeStatistics(ML_update_time,valid_time,t2m_obs,t2m_nwp,t2m_pred):
    """
    Calculate different statistics measures for LEAD time.
    """

    # Dimensions
    nstation = len(t2m_pred)
    nlead_time = t2m_pred[0].shape[1]

    # Define statistics arrays
    stat_pred = np.full((nstation,nlead_time), fill_value=np.nan, dtype=stat_array_type_list)
    stat_nwp  = np.full((nstation,nlead_time), fill_value=np.nan, dtype=stat_array_type_list)
    nobs_pred = np.full((nstation,nlead_time), fill_value=np.nan, dtype=int)
    nobs_nwp  = np.full((nstation,nlead_time), fill_value=np.nan, dtype=int)

    Nmin = 60

    for istation in range(nstation):
        for ilead in range(nlead_time):
            step = int(fc_update/fc_output_interval)
            mask_nwp = ( (ML_update_time[istation][::step,ilead] < valid_time[istation][::step,ilead]) )
            mask_pred = ( (ML_update_time[istation][:,ilead] < valid_time[istation][:,ilead]) )

            t2m_nwp_lead = t2m_nwp[istation][::step,ilead][mask_nwp].flatten()
            t2m_nwp_obs_lead = t2m_obs[istation][::step,ilead][mask_nwp].flatten()
            t2m_pred_lead  = t2m_pred[istation][:,ilead][mask_pred].flatten()
            t2m_pred_obs_lead = t2m_obs[istation][:,ilead][mask_pred].flatten()

            nobs_pred[istation,ilead] = np.sum(~np.isnan(t2m_pred_lead-t2m_pred_obs_lead))
            nobs_nwp[istation,ilead] = np.sum(~np.isnan(t2m_nwp_lead-t2m_nwp_obs_lead))
            if ( (nobs_pred[istation,ilead] > Nmin) & (nobs_nwp[istation,ilead] > Nmin) ):
                stat_pred[:][istation,ilead] = statistics_measures(t2m_pred_lead,t2m_pred_obs_lead,stat_pred[:][istation,ilead])
                stat_nwp[:][istation,ilead] = statistics_measures(t2m_nwp_lead,t2m_nwp_obs_lead,stat_nwp[:][istation,ilead])

    return stat_nwp, stat_pred, nobs_nwp, nobs_pred


def CalculateSeasonalLeadTimeStatistics(ML_update_time,valid_time,valid_datetime,t2m_obs,t2m_nwp,t2m_pred):
    """
    Calculate different statistics measures for LEAD time.
    """

    # Dimensions
    nstation = len(t2m_pred)
    nlead_time = t2m_pred[0].shape[1]

    # Define statistics arrays
    stat_pred = np.full((nstation,nseasons,nlead_time), fill_value=np.nan, dtype=stat_array_type_list)
    stat_nwp  = np.full((nstation,nseasons,nlead_time), fill_value=np.nan, dtype=stat_array_type_list)
    nobs_pred = np.full((nstation,nseasons,nlead_time), fill_value=np.nan, dtype=int)
    nobs_nwp  = np.full((nstation,nseasons,nlead_time), fill_value=np.nan, dtype=int)

    Nmin = 60

    for istation in range(nstation):
        for iseason in range(nseasons):
            for ilead in range(nlead_time):
                step = int(fc_update/fc_output_interval)
                mask_nwp = ( ( ML_update_time[istation][::step,ilead] < \
                               valid_time[istation][::step,ilead] ) & \
                             ( ( pd.DatetimeIndex(valid_datetime[istation][::step,ilead]).month == seasons[iseason][0] ) | \
                               ( pd.DatetimeIndex(valid_datetime[istation][::step,ilead]).month == seasons[iseason][1] ) | \
                               ( pd.DatetimeIndex(valid_datetime[istation][::step,ilead]).month == seasons[iseason][2] ) ) )
                mask_pred = ( ( ML_update_time[istation][:,ilead] < valid_time[istation][:,ilead] ) & \
                              ( ( pd.DatetimeIndex(valid_datetime[istation][:,ilead]).month == seasons[iseason][0] ) | \
                                ( pd.DatetimeIndex(valid_datetime[istation][:,ilead]).month == seasons[iseason][1] ) | \
                                ( pd.DatetimeIndex(valid_datetime[istation][:,ilead]).month == seasons[iseason][2] ) ) )
    
                t2m_nwp_lead = t2m_nwp[istation][::step,ilead][mask_nwp].flatten()
                t2m_nwp_obs_lead = t2m_obs[istation][::step,ilead][mask_nwp].flatten()
                t2m_pred_lead  = t2m_pred[istation][:,ilead][mask_pred].flatten()
                t2m_pred_obs_lead = t2m_obs[istation][:,ilead][mask_pred].flatten()
    
                nobs_pred[istation,iseason,ilead] = np.sum(~np.isnan(t2m_pred_lead-t2m_pred_obs_lead))
                nobs_nwp[istation,iseason,ilead] = np.sum(~np.isnan(t2m_nwp_lead-t2m_nwp_obs_lead))
                if ( (nobs_pred[istation,iseason,ilead] > Nmin) & (nobs_nwp[istation,iseason,ilead] > Nmin) ):
                    stat_pred[:][istation,iseason,ilead] = statistics_measures(t2m_pred_lead,t2m_pred_obs_lead,stat_pred[:][istation,iseason,ilead])
                    stat_nwp[:][istation,iseason,ilead] = statistics_measures(t2m_nwp_lead,t2m_nwp_obs_lead,stat_nwp[:][istation,iseason,ilead])

    return stat_nwp, stat_pred, nobs_nwp, nobs_pred




def CalculatePooledLeadTimeStatistics(stat_nwp,stat_pred,nobs_nwp,nobs_pred):
    """
    Calculate the pooled station statistics
    """
    # Dimensions
    nlead_time = stat_nwp['ME'].shape[1]

    # Define statistics arrays
    stat_nwp_pool = np.full((nlead_time), fill_value=np.nan, dtype=stat_array_type_list)
    stat_pred_pool = np.full((nlead_time), fill_value=np.nan, dtype=stat_array_type_list)
    nobs_nwp_pool = np.full((nlead_time), fill_value=0, dtype=np.int)
    nobs_pred_pool = np.full((nlead_time), fill_value=0, dtype=np.int)

    for stat in stat_names:
        if (stat == 'ME'):
            stat_nwp_pool[stat] = np.nanmean(np.abs(stat_nwp[stat]), axis=0)
            stat_pred_pool[stat] = np.nanmean(np.abs(stat_pred[stat]), axis=0)
        else:
            stat_nwp_pool[stat] = np.nanmean(stat_nwp[stat], axis=0)
            stat_pred_pool[stat] = np.nanmean(stat_pred[stat], axis=0)

    nobs_nwp_pool = np.sum(nobs_nwp, axis=0)
    nobs_pred_pool = np.sum(nobs_pred, axis=0)

    return stat_nwp_pool, stat_pred_pool, nobs_nwp_pool, nobs_pred_pool



def CalculateSeasonalPooledLeadTimeStatistics(stat_nwp,stat_pred,nobs_nwp,nobs_pred):
    """
    Calculate the pooled station statistics per season
    """
    # Dimensions
    nlead_time = stat_nwp['ME'].shape[2]

    # Define statistics arrays
    stat_nwp_pool = np.full((nseasons,nlead_time), fill_value=np.nan, dtype=stat_array_type_list)
    stat_pred_pool = np.full((nseasons,nlead_time), fill_value=np.nan, dtype=stat_array_type_list)
    nobs_nwp_pool = np.full((nseasons,nlead_time), fill_value=0, dtype=np.int)
    nobs_pred_pool = np.full((nseasons,nlead_time), fill_value=0, dtype=np.int)

    for iseason in range(nseasons):
        for stat in stat_names:
            if (stat == 'ME'):
                stat_nwp_pool[stat][iseason,:] = np.nanmean(np.abs(stat_nwp[stat][:,iseason,:]), axis=0)
                stat_pred_pool[stat][iseason,:] = np.nanmean(np.abs(stat_pred[stat][:,iseason,:]), axis=0)
            else:
                stat_nwp_pool[stat][iseason,:] = np.nanmean(stat_nwp[stat][:,iseason,:], axis=0)
                stat_pred_pool[stat][iseason,:] = np.nanmean(stat_pred[stat][:,iseason,:], axis=0)


        nobs_nwp_pool[iseason,:] = np.sum(nobs_nwp[:,iseason,:], axis=0)
        nobs_pred_pool[iseason,:] = np.sum(nobs_pred[:,iseason,:], axis=0)

    return stat_nwp_pool, stat_pred_pool, nobs_nwp_pool, nobs_pred_pool



def PlotStationDatasetMap(lat,lon,dataset_name,Zoom_DK,Save_Plot,fig_dir):
    """
    Plot lat,lon of stations on a map
    """
    # Dimensions
    nstation = len(lat)

    if dataset_name == 'Train':
        color = 'g'
        marker = 'o'
        dataset_title = 'Training dataset'
    elif dataset_name == 'Test':
        color = 'r'
        marker = '^'
        dataset_title = 'Test dataset'
    elif dataset_name == 'Validation':
        color = 'b'
        marker = '*'
        dataset_title = 'Validation dataset'

    #==========================
    # Map of station locations
    #==========================
    if Zoom_DK:
        # Map coordinates
        lon_min = 7
        lon_max = 16
        lat_min = 54
        lat_max = 58
    
        # Tick coordinates
        lon_tick_min = 5
        lon_tick_max = 20
        dlon_tick = 1
        lat_tick_min = 50
        lat_tick_max = 60
        dlat_tick = 1
    else:
        # Map coordinates
        lon_min = 3
        lon_max = 30
        lat_min = 50
        lat_max = 65
    
        # Tick coordinates
        lon_tick_min = 0
        lon_tick_max = 30
        dlon_tick = 5
        lat_tick_min = 50
        lat_tick_max = 70
        dlat_tick = 5


    # Ticks
    lon_ticks = np.arange(lon_tick_min,lon_tick_max,dlon_tick)
    lat_ticks = np.arange(lat_tick_min,lat_tick_max,dlat_tick)

    # Figure
    fig, ax = plt.subplots(figsize=[8, 7])
    m = basemap_subplot(ax,lat_min,lat_max,lon_min,lon_max,lat_ticks,lon_ticks)
 
    
    # Plot station positions
    x, y = m(lon, lat)
    for i in range(nstation):
        m.scatter(x[i], y[i], marker=marker, color=color, s=100, zorder=2)

    ax.set_title(dataset_title, fontsize=20, weight='bold')

    if Save_Plot:
        fig_file = fig_dir + dataset_name + '_Locations_' + str(nstation) + '.png'
        plt.savefig(fig_file)
        plt.close()
    else:
        plt.show()




def PlotStationSeasonalStatisticsMap(stat_nwp,stat_pred,lat,lon,fig_dir,Save_Plot,Zoom_DK,Validation_Labels):
    """
    Plot the pooled station statistics map
    """
    # Dimensions
    nstation = len(lat)

    # Combine into one list
    stats = [stat_nwp, stat_pred]

    # Color map settings
    GHRSSTanomalycolor = setGHRSSTanomalyColor()
    GHRSSTpositiveanomalycolor = setGHRSSTpositiveanomalyColor()
    color_map = [GHRSSTanomalycolor, GHRSSTpositiveanomalycolor, GHRSSTpositiveanomalycolor, \
                 GHRSSTpositiveanomalycolor, GHRSSTpositiveanomalycolor, GHRSSTpositiveanomalycolor]

    # Colorbar settings
    cmin = [-2., 0., 0., 0., 0., 0.8]
    cmax = [2., 2., 2.5, 2., 3., 1.]

    # Label settings
    cb_labels = ['Bias ($^{\circ}$C)','MAE ($^{\circ}$C)','STD ($^{\circ}$C)', \
                 'STD MAE ($^{\circ}$C)','RMSE ($^{\circ}$C)','Correlation ($^{\circ}$C)']
    pp_methods = ['GFS','Transformer']
    if ['HARMONIE_data']:
        pp_methods = ['GFS','Transformer','HARMONIE']

    if Zoom_DK:
        # Map coordinates
        lon_min = 7
        lon_max = 16
        lat_min = 54
        lat_max = 58
    
        # Tick coordinates
        lon_tick_min = 5
        lon_tick_max = 20
        dlon_tick = 1
        lat_tick_min = 50
        lat_tick_max = 60
        dlat_tick = 1
    else:
        # Map coordinates
        lon_min = 3
        lon_max = 30
        lat_min = 50
        lat_max = 65
    
        # Tick coordinates
        lon_tick_min = 0
        lon_tick_max = 30
        dlon_tick = 5
        lat_tick_min = 50
        lat_tick_max = 70
        dlat_tick = 5

    # Ticks
    lon_ticks = np.arange(lon_tick_min,lon_tick_max,dlon_tick)
    lat_ticks = np.arange(lat_tick_min,lat_tick_max,dlat_tick)


    nrows = 1
    ncols = 2
    widths = [5,5,0.4]

    # Loop over statistics
    for istat,stat_name in enumerate(stat_names):
        for iseason in range(nseasons):
            # Figure and axes
            fig = plt.figure(figsize=[15, 5])
            gs = fig.add_gridspec(nrows, ncols+1, width_ratios=widths)
    
            axes = []
            images = []
            for i in range(len(stats)):
                # 2D indices
                ix = i % ncols
                iy = i // ncols
    
                # Get gridspec ax
                ax = fig.add_subplot(gs[iy,ix])
                # Add basemap projection
                m = basemap_subplot(ax,lat_min,lat_max,lon_min,lon_max,lat_ticks,lon_ticks)
    
                # Plot station positions with stat as colorbar
                x, y = m(lon, lat)
                points = m.scatter(x, y, marker='o', s=30, c=stats[i][stat_name][iseason,:], cmap=color_map[istat], zorder=2)  # FieldSense
                images.append(points)
                if Validation_Labels:
                    for istation in range(nstation):
                        plt.text(x[istation],y[istation],str(istation+1),color='k')
                # Set title
                ax.set_title(pp_methods[i], fontsize=14, weight='bold')
                axes.append(ax)
    
            # Colorbar min and max
            vmin = cmin[istat]
            vmax = cmax[istat]
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            for im in images:
                im.set_norm(norm)
            # Shared colorbar
            axes.append(fig.add_subplot(gs[:,ncols]))
            cb = fig.colorbar(points, cax=axes[-1])
            cb.set_label(label=cb_labels[istat], fontsize=14)
            cb.ax.tick_params(labelsize=14)

            # Overall title
            plt.suptitle(season_labels[iseason], fontsize=14, weight='bold')
    
            if Save_Plot:
                fig_file = fig_dir + 'Season_' + season_name[iseason] + '_Map_Stat_' + stat_name + '.png'
                plt.savefig(fig_file)
                plt.close()
            else:
                plt.show()





def PlotPooledLeadTimeStatistics(x,stat_nwp,stat_pred,Save_Plot,fig_dir):
    """
    Plot the pooled lead time statistics with BIAS and STD in the same plot
    """

    stat_plot_names = ['ME', 'STD_ME']
    stat_labels = ['mean', 'std']
    stat_ylabel = 'T$_{2m, fcst}$ - T$_{2m, obs}$ '
    linestyles = ['-','--']

    # Xticks
    xmin = 0; xmax = 49; dx = 1
    Xticks = np.arange(xmin,xmax+dx,dx)
    Xticks = Xticks[::5]
    Xticklabels = Xticks
    ymin = 0.
    ymax = 2.0
    dy = 0.2
    Yticks = np.arange(ymin,ymax+dy,dy)
    Yticklabels = Yticks

    fig, ax = plt.subplots(figsize=[8,3.5])
    for istat,stat in enumerate(stat_plot_names):
        ax.plot(x,stat_nwp[stat], color='r', linestyle=linestyles[istat], linewidth=2., label='GFS ' + stat_labels[istat])
        ax.plot(x,stat_pred[stat], color='b', linestyle=linestyles[istat], linewidth=2., label='Transformer ' + stat_labels[istat])

    ax.legend()
    ax.grid(True, axis='both')

    ax.set_xlabel('Forecast lead time (h)', fontsize=16)#, labelpad=15.)
    ax.set_ylabel(stat_ylabel + ' ($^{\circ}$C)', fontsize=16, labelpad=15.)
    ax.set_xticks(Xticks)
    ax.set_yticks(Yticks)
    ax.set_xticklabels(Xticklabels, fontsize=11)
    ax.set_yticklabels(Yticklabels, fontsize=11)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    
    ax.legend(bbox_to_anchor=(1.01,1.), loc="upper left", fontsize='10')
#    ax.legend(loc="upper left", fontsize='14')
#    ax.legend(loc="best", fontsize='14')

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    
    fig.tight_layout()

    if Save_Plot:
        fig_file = fig_dir + 'LeadtimeStatistics.png'
        plt.savefig(fig_file)
        plt.close()
    else:
        plt.show()



def PlotSeasonalPooledLeadTimeStatistics(x,stat_nwp,stat_pred,Save_Plot,fig_dir):
    """
    Plot the pooled seasonal lead time statistics with BIAS and STD in the same plot
    """
    stat_plot_names = ['ME', 'STD_ME']
    stat_labels = ['mean', 'std']
    stat_ylabel = 'T$_{2m, fcst}$ - T$_{2m, obs}$ '
    linestyles = ['-','--']

    # Xticks
    xmin = 0; xmax = 49; dx = 1
    Xticks = np.arange(xmin,xmax+dx,dx)
    Xticks = Xticks[::5]
    Xticklabels = Xticks
    ymin = 0.
    ymax = 2.2#2.0
    dy = 0.2
    Yticks = np.arange(ymin,ymax+dy,dy)
    Yticklabels = Yticks


    for iseason in range(nseasons):
        fig, ax = plt.subplots(figsize=[8,3.5])
        for istat,stat in enumerate(stat_plot_names):
            ax.plot(x,stat_nwp[stat][iseason], color='r', linestyle=linestyles[istat], linewidth=2., label='GFS ' + stat_labels[istat])
            ax.plot(x,stat_pred[stat][iseason], color='b', linestyle=linestyles[istat], linewidth=2., label='Transformer ' + stat_labels[istat])

        ax.legend()
        ax.grid(True, axis='both')

        ax.set_xlabel('Forecast lead time (h)', fontsize=16)#, labelpad=15.)
        ax.set_ylabel(stat_ylabel + ' ($^{\circ}$C)', fontsize=16, labelpad=15.)
        ax.set_xticks(Xticks)
        ax.set_yticks(Yticks)
        ax.set_xticklabels(Xticklabels, fontsize=11)
        ax.set_yticklabels(Yticklabels, fontsize=11)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        
        ax.set_title(season_labels[iseason], fontsize=16, weight='bold')

        ax.legend(bbox_to_anchor=(1.01,1.), loc="upper left", fontsize='10')
#        ax.legend(loc="upper left", fontsize='14')
#        ax.legend(loc="best", fontsize='14')

        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        
        fig.tight_layout()


        if Save_Plot:
            fig_file = fig_dir + 'Season_' + season_name[iseason] + '_LeadtimeStatistics.png'
            plt.savefig(fig_file)
            plt.close()
        else:
            plt.show()





def setGHRSSTanomalyColor():
    """
    Create the GHRSTT anomaly color map
    """

    cmap = np.array([ [ 107,   0, 219],
                      [ 122,   0, 213],
                      [ 138,   0, 208],
                      [ 156,   0, 201],
                      [ 131,  24, 209],
                      [  85,  60, 225],
                      [  39,  97, 241],
                      [   0, 133, 255],
                      [   0, 169, 255],
                      [   0, 211, 255],
                      [   0, 247, 255],
                      [  29, 255, 226],
                      [  65, 255, 190],
                      [ 102, 255, 154],
                      [ 133, 255, 131],
                      [ 154, 255, 141],
                      [ 173, 255, 150],
                      [ 191, 255, 159],
                      [ 192, 238, 168],
                      [ 191, 220, 177],
                      [ 191, 202, 186],
                      [ 202, 202, 183],
                      [ 220, 220, 168],
                      [ 238, 238, 154],
                      [ 255, 254, 137],
                      [ 255, 236,  97],
                      [ 255, 218,  58],
                      [ 255, 197,  11],
                      [ 255, 179,   0],
                      [ 255, 161,   0],
                      [ 255, 142,   0],
                      [ 255, 120,   0],
                      [ 255,  84,   0],
                      [ 255,  41,   0],
                      [ 255,   5,   0],
                      [ 246,   0,  37],
                      [ 236,   0,  79],
                      [ 227,   0, 122],
                      [ 211,   0, 135],
                      [ 180,   0,  85],
                      [ 154,   0,  43],
                      [ 128,   0,   0] ])/255

    GHRSSTanomalyColorMap = LinearSegmentedColormap.from_list('GHRSSTanomalyColor', cmap)

    return GHRSSTanomalyColorMap



def setGHRSSTpositiveanomalyColor():
    """
    Create the GHRSTT positive anomaly color map
    """

    cmap = np.array([ [ 202, 202, 183],
                      [ 220, 220, 168],
                      [ 238, 238, 154],
                      [ 255, 254, 137],
                      [ 255, 236,  97],
                      [ 255, 218,  58],
                      [ 255, 197,  11],
                      [ 255, 179,   0],
                      [ 255, 161,   0],
                      [ 255, 142,   0],
                      [ 255, 120,   0],
                      [ 255,  84,   0],
                      [ 255,  41,   0],
                      [ 255,   5,   0],
                      [ 246,   0,  37],
                      [ 236,   0,  79],
                      [ 227,   0, 122],
                      [ 211,   0, 135],
                      [ 180,   0,  85],
                      [ 154,   0,  43],
                      [ 128,   0,   0] ])/255

    GHRSSTpositiveanomalyColorMap = LinearSegmentedColormap.from_list('GHRSSTpositiveanomalyColor', cmap)

    return GHRSSTpositiveanomalyColorMap

