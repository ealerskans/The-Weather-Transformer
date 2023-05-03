#==================
# Library imports
#==================
import numpy as np
import os


# Paths
data_dir = '/media/emy/Elements SE/Emy/Projects/PhD/Data/the-weather-transformer/'
nwp_data_dir = data_dir + 'nwp_data/'
station_data_dir = data_dir + '/obs_data/'
project_dir = os.getcwd()
fig_dir = project_dir + '/figs/'
list_dir = project_dir + '/lists/'
model_dir = project_dir + '/models/'

# General
pi = np.pi

# Time
hours_per_day = 24

# Conversions
kelvin2celsius = -273.15
celsius2kelvin = 273.15
rad2deg = 180/pi
deg2rad = pi/180

# netCDF fil values
fill_value_int = -9999
fill_value_float = -9999.

# Number of stations to train, validation and test on
nstation_train = 4
nstation_val = 3
nstation_test = 3

# Forecast settings
fc_len = 54                # Number of forecast lead times/forecast length (in hours)
fc_output_interval = 3     # Output interval of the NWP data (in hours)
fc_update = 6              # How often the forecast is updated (in hours)
nlead_time = int((fc_len-1)/fc_output_interval) + 1
nfc_cycles = int(24/fc_update)
nvalid_time = int(24/fc_output_interval)

# Format settings
fmt_strp = '%Y-%m-%dT%H:%M:%S.%fZ'
fmt_strf = '%Y%m%d'
fmt_out = '%Y%m%d%H%M%S'

# Prediction settings
input_days = 5                                # Temporal extent of the model, i.e. how much data we give the model,
                                              # from which it should make predictions
block_size = input_days*(hours_per_day//3)
nfc_input = input_days*nfc_cycles
prediction_window = nlead_time - 2            # Prediction window


# ML parameters
n_epochs = 5#200
batch_size = 128#384
loss_metrics = ['MSE']


