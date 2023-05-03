"""
NAME:
         general

PURPOSE:
         Definitions for general functions and global parameters

CONTAINS:

"""

#=========================
# Standard library imports
#=========================
import argparse
import ast
import datetime
import glob
import h5py
import numpy as np
import os
import pickle
import sys


#================
# Local imports
#================
from tools import does_directory_exist, create_dir



def GetDataVariables(data_type):
    """
    Get variable names and types.

    Input:
    + data_type: Specifier for which names and type to extract
                  nwp: NWP data
                  obs: OBS data
    """

    if (data_type == 'nwp'):
        var_names = ['issue_unix','valid_unix','imei','lat','lon','t2m','rh2m','wspd10m','u10m','v10m', 'q2m', \
                     'wsdir10m','td2m','mslp','lhtfl','shtfl','nswrf','nlwrf','tcc','pwat','gh500','gh700','gh850', \
                     't500','t700','t850','u500','u700','u850','v500','v700','v850','w700','tp']
        var_types = ['f8','f8','i8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8', \
                     'f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8']
    elif (data_type == 'obs'):
        var_names = ['unix','datetime','imei','lat','lon','t2m', 'wind_speed','pres','precp']
        var_types = ['f8','datetime64[s]','i8','f8','f8','f8','f8','f8','f8']
    elif (data_type == 'stat'):
        var_names = ['ME', 'MAE','STD_ME', 'STD_MAE','RMSE', 'r']
        var_types = ['f8', 'f8', 'f8', 'f8', 'f8', 'f8']
    elif (data_type == 'analysis'):
        var_names = ['issue_unix','valid_unix']
        var_types = ['f8','f8',]
    else:
        print('Could not handle the specified data_type: ' + data_type)
        sys.exit()


    return var_names, var_types




def InitializeFeatures(args):
    """
    Initialize the features to be used
    """

    # Input features
    if len(args) > 1:
        print('Input features from command-line arguments')
        # Convert string representation of dictionary to dictionary
        include_features = ast.literal_eval(args[1])
    else:
        print('Pre-defined input features')
        include_features = {}
        include_features['OBS_t2m'] = True
        include_features['NWP_t2m'] = True
        include_features['NWP_u10m'] = False
        include_features['NWP_v10m'] = False
        include_features['NWP_wspd10m'] = True
        include_features['NWP_sin_wsdir10m'] = True
        include_features['NWP_cos_wsdir10m'] = True
        include_features['NWP_rh2m'] = True
        include_features['NWP_q2m'] = True
        include_features['NWP_td2m'] = True
        include_features['NWP_mslp'] = True
        include_features['NWP_lhtfl'] = False
        include_features['NWP_shtfl'] = True
        include_features['NWP_nswrf'] = True
        include_features['NWP_nlwrf'] = False
        include_features['NWP_tcc'] = False
        include_features['NWP_tp'] = False
        include_features['NWP_pwat'] = True
        include_features['NWP_gh500'] = True
        include_features['NWP_gh700'] = True
        include_features['NWP_gh850'] = True
        include_features['NWP_t500'] = False
        include_features['NWP_t700'] = False
        include_features['NWP_t850'] = False
        include_features['NWP_u500'] = False
        include_features['NWP_u700'] = False
        include_features['NWP_u850'] = False
        include_features['NWP_v500'] = False
        include_features['NWP_v700'] = False
        include_features['NWP_v850'] = False
        include_features['NWP_w700'] = False
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

    return include_features, feature_indices



def SpecifyFeatures(include_features):
    """
    Create the list of features and specify the run directory for logging training events

    Input:
    + include_features: Dictionary with features to include

    Output:
    + features: List of features to include in the model
    + Nfeatures: Number of features to include in the model
    """
    # List of features
    features = []

    if include_features['OBS_t2m']:
        features.append('t2m_obs')
    if include_features['NWP_t2m']:
        features.append('t2m_nwp')
    if include_features['NWP_u10m']:
        features.append('u10m_nwp')
    if include_features['NWP_v10m']:
        features.append('v10m_nwp')
    if include_features['NWP_wspd10m']:
        features.append('wspd10m_nwp')
    if include_features['NWP_sin_wsdir10m']:
        features.append('sin_wsdir10m_nwp')
    if include_features['NWP_cos_wsdir10m']:
        features.append('cos_wsdir10m_nwp')
    if include_features['NWP_rh2m']:
        features.append('rh2m_nwp')
    if include_features['NWP_q2m']:
        features.append('q2m_nwp')
    if include_features['NWP_td2m']:
        features.append('td2m_nwp')
    if include_features['NWP_mslp']:
        features.append('mslp_nwp')
    if include_features['NWP_lhtfl']:
        features.append('lhtfl_nwp')
    if include_features['NWP_shtfl']:
        features.append('shtfl_nwp')
    if include_features['NWP_nswrf']:
        features.append('nswrf_nwp')
    if include_features['NWP_nlwrf']:
        features.append('nlwrf_nwp')
    if include_features['NWP_tcc']:
        features.append('tcc_nwp')
    if include_features['NWP_tp']:
        features.append('tp_nwp')
    if include_features['NWP_pwat']:
        features.append('pwat_nwp')
    if include_features['NWP_gh500']:
        features.append('gh500_nwp')
    if include_features['NWP_gh700']:
        features.append('gh700_nwp')
    if include_features['NWP_gh850']:
        features.append('gh850_nwp')
    if include_features['NWP_t500']:
        features.append('t500_nwp')
    if include_features['NWP_t700']:
        features.append('t700_nwp')
    if include_features['NWP_t850']:
        features.append('t850_nwp')
    if include_features['NWP_u500']:
        features.append('u500_nwp')
    if include_features['NWP_u700']:
        features.append('u700_nwp')
    if include_features['NWP_u850']:
        features.append('u850_nwp')
    if include_features['NWP_v500']:
        features.append('v500_nwp')
    if include_features['NWP_v700']:
        features.append('v700_nwp')
    if include_features['NWP_v850']:
        features.append('v850_nwp')
    if include_features['NWP_w700']:
        features.append('w700_nwp')
    if include_features['TIME_sin_doy']:
        features.append('sin_doy_time')
    if include_features['TIME_cos_doy']:
        features.append('cos_doy_time')
    if include_features['TIME_sin_hod']:
        features.append('sin_hod_time')
    if include_features['TIME_cos_hod']:
        features.append('cos_hod_time')
    if include_features['NWP_wspd500']:
        features.append('wspd500_nwp')
    if include_features['NWP_wspd700']:
        features.append('wspd700_nwp')
    if include_features['NWP_wspd850']:
        features.append('wspd850_nwp')
    if include_features['NWP_sin_wsdir500']:
        features.append('sin_wsdir500_nwp')
    if include_features['NWP_cos_wsdir500']:
        features.append('cos_wsdir500_nwp')
    if include_features['NWP_sin_wsdir700']:
        features.append('sin_wsdir700_nwp')
    if include_features['NWP_cos_wsdir700']:
        features.append('cos_wsdir700_nwp')
    if include_features['NWP_sin_wsdir850']:
        features.append('sin_wsdir850_nwp')
    if include_features['NWP_cos_wsdir850']:
        features.append('cos_wsdir850_nwp')


    Nfeatures = len(features)


    return features, Nfeatures



def SpecifyDatasetFile(data_dir,include_features,nstation_train,nstation_val,nstation_test,block_size_days):
    """
    Specify name for output files containing the datasets
    """

    # Model name
    datasets_file_generic = data_dir + 'Dataset_Generic'
    datasets_file_generic += '_nstation.' + str(nstation_train) + '.' + str(nstation_val) + '.' + str(nstation_test) + '.hdf5'
    datasets_file_train = data_dir + 'Dataset_Train'
    datasets_file_train += '_nstation.' + str(nstation_train) + '.' + str(nstation_val) + '.' + str(nstation_test) + '.hdf5'
    datasets_file_val = data_dir + 'Dataset_Validation'
    datasets_file_val += '_nstation.' + str(nstation_train) + '.' + str(nstation_val) + '.' + str(nstation_test) + '.hdf5'
    datasets_file_test = data_dir + 'Dataset_Test'
    datasets_file_test += '_nstation.' + str(nstation_train) + '.' + str(nstation_val) + '.' + str(nstation_test) + '.hdf5'


    return datasets_file_generic, datasets_file_train, datasets_file_val, datasets_file_test



def SpecifyRun(project_dir,Save_Run,features,nstation_train,nstation_test,n_epochs, \
               batch_size,block_size_days,loss_metrics):
    """
    Specify (and create if not existing already) the run directory for logging training events

    Input:
    + project_dir: Project directory
    + Save_Run: Logical for saving the run output (True) or not (False)
    + features: Feature names of features to include
    + nstation_train: Number of stations used for training
    + nstation_test: Number of stations used for testing
    + n_epochs: Number of epochs
    + batch_size: Batch size
    + block_size_days: Size of input data (in days)
    + loss_metrics: Loss metrics to evaluate performance of the model on

    Output:
    + run_dir: Run directory for logging training events
    (+ README: README file containing info on the run - a file outputted in run_dir)
    """

    # Directory for current run
    run_dir = project_dir + '/runs/'

    if Save_Run:
        # Get all directories on the format "run.Exp-XXX"
        files = glob.glob(run_dir + 'run.Exp-*')
        files = [ifile[len(run_dir):] for ifile in files]
        if not files:
           run_number = 0
        elif (len(files) == 1):
            run_number = 1
        else:
            run_number = np.max([int(ifile[-3:].strip('0')) if ifile[-3:] != '000' else 0 for ifile in files]) + 1
    
        # Run directory
        run_dir += 'run.Exp-' + str(run_number).zfill(3)
        # Check if run_dir exists, if not Create the run directory
        if not does_directory_exist(run_dir):
            create_dir(run_dir)

        # Write run information to README file
        ofile = run_dir + '/README'
        content  = 'README file containing information on run: run.Exp-' + str(run_number).zfill(3) + ':'
        # - OBS variables
        content += '\nOBS variables:'
        obs_features = [feature.strip('_obs') for feature in features if 'obs' in feature]
        for feature in obs_features:
            content += '\n+ ' + feature
        # - NWP variables
        content += '\nNWP variables:'
        nwp_features = [feature.strip('_nwp') for feature in features if 'nwp' in feature]
        for feature in nwp_features:
            content += '\n+ ' + feature
        # - Number of stations used for train and test 
        content += '\nNumber of stations used for (1) train and (2) test:'
        content += '\n(1). ' + str(nstation_train)
        content += '\n(2). ' + str(nstation_test)
        # - Block size
        content += '\nBlock size (aka input length) = ' + str(block_size_days) + ' days'
        # - Batch size
        content += '\nBatch size = ' + str(batch_size)
        # - Number of epochs
        content += '\nNumber of epochs = ' + str(n_epochs)
        # - Othe comments/notes
        content += '\nComments/Notes:'
        content += '\nInterpolated missing input data'
        content += '\nLoss metrics - '
        for i,loss in enumerate(loss_metrics):
            if i == 0:
                content += loss
            else:
                content += ' + ' + loss
    
        # Write the output to the README file
        write_file = open(ofile, 'w')
        n = write_file.write(content)
        write_file.close()


    else:
        run_dir += 'test_run'


    return run_dir



def SpecifyModel(model_dir,include_features,features,nstation_train,nstation_val,n_epochs, \
                 batch_size,block_size_days,loss_metrics):
    """
    Specify model name

    Input:
    + model_dir: Model directory
    + include_features: Dictionary with features to include
    + features: Feature names of features to include
    + nstation_train: Number of stations used for training
    + nstation_val: Number of stations used for validation

    Output:
    + model_name: Name of the trained model
    """

    # File prefix
    file_prefix = model_dir + 't2m_minGPT'
    # Specify file
    model_file = SpecifyFile(file_prefix,include_features,nstation_train,nstation_val, \
                             n_epochs,batch_size,block_size_days,loss_metrics)

    return model_file



def SpecifyPredictionFile(data_dir,include_features,nstation_train,nstation_val,nstation_test,n_epochs, \
                          batch_size,block_size_days,loss_metrics):
    """
    Specify name for output files containing the predictions
    """

    # File prefix
    file_prefix = data_dir + 'T2m_Predictions'
    # Specify file
    predictions_file = SpecifyFile(file_prefix,include_features,nstation_train,nstation_val,n_epochs,batch_size, \
                                   block_size_days,loss_metrics,include_test=True,nstation_test=nstation_test)
    # Add extension
    predictions_file += '.pickle'

    return predictions_file



def SpecifyFile(file_prefix,include_features,nstation_train,nstation_val,n_epochs,batch_size, \
                block_size_days,loss_metrics,include_test=False,nstation_test=0):
    """
    Specify file name
    """

    file_name = ''
    if include_features['OBS_t2m']:
        file_name += '_OBS.t2m'
    if include_features['NWP_t2m']:
        file_name = file_name + '.t2m' if 'NWP' in file_name else file_name + '_NWP.t2m'
    if include_features['NWP_u10m']:
        file_name = file_name + '.u10m' if 'NWP' in file_name else file_name + '_NWP.u10m'
    if include_features['NWP_v10m']:
        file_name = file_name + '.v10m' if 'NWP' in file_name else file_name + '_NWP.v10m'
    if ( include_features['NWP_wspd10m'] & include_features['NWP_sin_wsdir10m'] & include_features['NWP_cos_wsdir10m']):
        file_name = file_name + '.wspd_phi10m' if 'NWP' in file_name else file_name + '_NWP.wspd_phi10m'
    if include_features['NWP_rh2m']:
        file_name = file_name + '.rh2m' if 'NWP' in file_name else file_name + '_NWP.rh2m'
    if include_features['NWP_q2m']:
        file_name = file_name + '.q2m' if 'NWP' in file_name else file_name + '_NWP.q2m'
    if include_features['NWP_td2m']:
        file_name = file_name + '.td2m' if 'NWP' in file_name else file_name + '_NWP.td2m'
    if include_features['NWP_mslp']:
        file_name = file_name + '.mslp' if 'NWP' in file_name else file_name + '_NWP.mslp'
    if include_features['NWP_lhtfl']:
        file_name = file_name + '.lht' if 'NWP' in file_name else file_name + '_NWP.lht'
    if include_features['NWP_shtfl']:
        file_name = file_name + '.sht' if 'NWP' in file_name else file_name + '_NWP.sht'
    if include_features['NWP_nswrf']:
        file_name = file_name + '.nswrf' if 'NWP' in file_name else file_name + '_NWP.nswrf'
    if include_features['NWP_nlwrf']:
        file_name = file_name + '.nlwrf' if 'NWP' in file_name else file_name + '_NWP.nlwrf'
    if include_features['NWP_tcc']:
        file_name = file_name + '.tcc' if 'NWP' in file_name else file_name + '_NWP.tcc'
    if include_features['NWP_tp']:
        file_name = file_name + '.tp' if 'NWP' in file_name else file_name + '_NWP.tp'
    if include_features['NWP_pwat']:
        file_name = file_name + '.pwat' if 'NWP' in file_name else file_name + '_NWP.pwat'
    if include_features['NWP_gh500']:
        file_name = file_name + '.gh500' if 'NWP' in file_name else file_name + '_NWP.gh500'
    if include_features['NWP_gh700']:
        file_name = file_name + '.gh700' if 'NWP' in file_name else file_name + '_NWP.gh700'
    if include_features['NWP_gh850']:
        file_name = file_name + '.gh850' if 'NWP' in file_name else file_name + '_NWP.gh850'
    if include_features['NWP_t500']:
        file_name = file_name + '.t500' if 'NWP' in file_name else file_name + '_NWP.t500'
    if include_features['NWP_t700']:
        file_name = file_name + '.t700' if 'NWP' in file_name else file_name + '_NWP.t700'
    if include_features['NWP_t850']:
        file_name = file_name + '.t850' if 'NWP' in file_name else file_name + '_NWP.t850'
    if include_features['NWP_u500']:
        file_name = file_name + '.u500' if 'NWP' in file_name else file_name + '_NWP.u500'
    if include_features['NWP_u700']:
        file_name = file_name + '.u700' if 'NWP' in file_name else file_name + '_NWP.u700'
    if include_features['NWP_u850']:
        file_name = file_name + '.u850' if 'NWP' in file_name else file_name + '_NWP.u850'
    if include_features['NWP_v500']:
        file_name = file_name + '.v500' if 'NWP' in file_name else file_name + '_NWP.v500'
    if include_features['NWP_v700']:
        file_name = file_name + '.v700' if 'NWP' in file_name else file_name + '_NWP.v700'
    if include_features['NWP_v850']:
        file_name = file_name + '.v850' if 'NWP' in file_name else file_name + '_NWP.v850'
    if include_features['NWP_w700']:
        file_name = file_name + '.w700' if 'NWP' in file_name else file_name + '_NWP.w700'
    if (include_features['TIME_sin_doy'] & include_features['TIME_cos_doy']):
        file_name = file_name + '.doy' if 'NWP' in file_name else file_name + '_NWP.doy'
    if (include_features['TIME_sin_hod'] & include_features['TIME_cos_hod']):
        file_name = file_name + '.hod' if 'NWP' in file_name else file_name + '_NWP.hod'
    if ( include_features['NWP_wspd500'] & include_features['NWP_sin_wsdir500'] & include_features['NWP_cos_wsdir500'] ):
        file_name = file_name + '.wspd_phi500' if 'NWP' in file_name else file_name + '_NWP.wspd_phi500'
    if ( include_features['NWP_wspd700'] & include_features['NWP_sin_wsdir700'] & include_features['NWP_cos_wsdir700'] ):
        file_name = file_name + '.wspd_phi700' if 'NWP' in file_name else file_name + '_NWP.wspd_phi700'
    if ( include_features['NWP_wspd850'] & include_features['NWP_sin_wsdir850'] & include_features['NWP_cos_wsdir850'] ):
        file_name = file_name + '.wspd_phi850' if 'NWP' in file_name else file_name + '_NWP.wspd_phi850'


    metrics = '_LossMetrics'
    for loss in loss_metrics:
        metrics += '.' + loss

    if include_test:
        output_file = file_prefix + file_name + '_nstation.' + str(nstation_train) + '.' + str(nstation_val) + '.' + str(nstation_test) + \
                           '_Epochs.' + str(n_epochs) + '_Batchsize.' + str(batch_size) + \
                           '_Blocksizedays.' + str(block_size_days) + metrics
    else:
        output_file = file_prefix + file_name + '_nstation.' + str(nstation_train) + '.' + str(nstation_val) + \
                           '_Epochs.' + str(n_epochs) + '_Batchsize.' + str(batch_size) + \
                           '_Blocksizedays.' + str(block_size_days) + metrics

    return output_file



