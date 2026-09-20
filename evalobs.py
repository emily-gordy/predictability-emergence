"""evaluate observations to find theoretical GMT emergence level from historically observed SST"""
#%%
import sys
import os

sys.path.append(os.path.abspath('./predictability_emergence')) 

import DataMakar
import DataHolder
import buildmodel
import helpers

import importlib as imp
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import torch.optim as optim

import torch.optim.lr_scheduler as lr_scheduler
import time

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

import glob
import json
import pickle
from scipy.stats import percentileofscore
from statsmodels.stats.contingency_tables import mcnemar

from PIL import Image

import cartopy.crs as ccrs
import matplotlib as mpl
from matplotlib.colors import BoundaryNorm

mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.size'] = 12
mpl.rcParams['font.sans-serif']=['Verdana']

params = {"ytick.color": "k",
          "xtick.color": "k",
          "axes.labelcolor": "k",
          "axes.edgecolor": "k"}
plt.rcParams.update(params)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

#%%

ssplist = ["370"]

ssplistplot = ["hist","370"] # thank god i haven't flipflopped on ssp labeling! that would be so annoying

# set user parameters

experiment_era = [1980,2100]
baselineera = [1900,1950]
trainvaltest = [np.arange(25),np.arange(25,38),np.arange(38,50)]

experiment_era_obs = [1970,2025]
baselineera_obs = [1950,1980]

inputlength = 10
# data params
outres = 10
timerange = [1900,2100]
filefront = "MPI_"
modelfilefront = "MPI_recordtemp_"
inres = 4
inputvar = 'tos'
outputvar = 'tas'
obstimerange = [1940,2025]
data_dir = 'data'

seedlist = [62469869,
            71856281,
            47621498,
            10431957,
            50561320,
            72166634,
            18469465,
            92895735,
            57693846,
            22284750]

imp.reload(DataHolder)

params = {
    "outres": outres,
    "timerange": timerange,
    "filefront": filefront,
    "inres": inres,
    "inputvar":inputvar,
    "outputvar":outputvar,
    "seedlist": seedlist,
    "obstimerange":obstimerange,
    "data_dir":data_dir
}

ntrain = 25
nval = 13
test = np.arange(38,50)

n_best = 3

#get the data

AllData = DataHolder.MPIInputOutput_SSPlist(params,ssplist)

latvec = AllData.output_lat
lonvec = AllData.output_lon

device = 'mps'
imp.reload(DataHolder)
AllObs = DataHolder.ERA5InputOutput(params)

#%% some exp params

gmtvec = np.arange(-0.5,2.5,0.05)

# expout = "MPI_histrecord_"
expout = "MPI_recordtemp_"

outputavgtime = 10

dummylat = -10
dummylon = 10
# _,obsinputgmt,_,_ = AllObs.obs_histrecordmax_nochopend(experiment_era_obs,baselineera_obs,inputlength,outputavgtime,dummylat,dummylon)
_,obsinputgmt,_,_ = AllObs.obs_recordmax_withrecordmax_nochopend(experiment_era_obs,baselineera_obs,inputlength,outputavgtime,dummylat,dummylon)
lenobs = len(obsinputgmt)
regenerate = True

#%%

for ilat,lat in enumerate(AllObs.output_lat):

    print('working on lat ' + str(lat))

    predsoutfile = "predictions/"+ expout + "avgtime_"+str(outputavgtime)+"_obs_lat_"+str(lat)+"_perturbGMT.pkl"

    if (lat>=0) & (regenerate):
        print('regenerating')
        # _,obsinputgmt,_,_ = AllObs.obs_histrecordmax_nochopend(experiment_era_obs,baselineera_obs,inputlength,outputavgtime,lat,dummylon)
        _,obsinputgmt,_,_ = AllObs.obs_recordmax_withrecordmax_nochopend(experiment_era_obs,baselineera_obs,inputlength,outputavgtime,lat,dummylon)
        lenobs = len(obsinputgmt)
        regenerate = False
    
    predsout = np.empty((36,3,len(gmtvec),lenobs))+np.nan
    predsout_climo = np.empty((36,3,len(gmtvec)))+np.nan

    for ilon,lon in enumerate(AllObs.output_lon):

        metricsin = "metrics/"+modelfilefront+"avgtime_"+str(outputavgtime)+"_allssps_lat_"+str(lat)+"_lon_"+str(lon)+"_seed*.json"
        filelist = glob.glob(metricsin)

        if len(filelist)!=0:

            print('working on lon '+ str(lon))

            bestseeds = helpers.get_best_files(filelist,n_best)

            # NO CHOP END
            # obsinput,obsinputgmt,obsinputpriorrecord,obsoutput = AllObs.obs_histrecordmax_nochopend(experiment_era_obs,baselineera_obs,inputlength,outputavgtime,lat,lon)
            obsinput,obsinputgmt,obsinputpriorrecord,obsoutput = AllObs.obs_recordmax_withrecordmax_nochopend(experiment_era_obs,baselineera_obs,inputlength,outputavgtime,lat,lon)

            obsinput_t = torch.tensor(obsinput,dtype=torch.float32)
            obsinputgmt_t = torch.tensor(1,dtype=torch.float32)*torch.ones((len(obsinputgmt),1),dtype=torch.float32)
            obsinputpriorrecord_t = torch.tensor(obsinputpriorrecord[-1].squeeze(),dtype=torch.float32)*torch.ones((len(obsinputgmt),1),dtype=torch.float32)
            
            obsinput_climo = AllObs.cutinput
            obsinput_meaninput = np.mean(obsinput_climo,axis=0,keepdims=True)
            obsinput_meaninput_anom = torch.tensor(obsinput_meaninput-np.mean(obsinput_meaninput,axis=1,keepdims=True),dtype=torch.float32)

            obsinputvectors_dummy_t = torch.cat((obsinputgmt_t,obsinputpriorrecord_t),axis=-1)
            for iseed,seed in enumerate(bestseeds):
                # load the model

                loadfile = "models/"+modelfilefront+"avgtime_"+str(outputavgtime)+"_allssps_lat_"+str(lat)+"_lon_"+str(lon)+"_seed_"+str(seed)+".pt"
                cnn = buildmodel.CNNclassifier(obsinput_t, obsinputvectors_dummy_t, 2).to('cpu')
                cnn.load_state_dict(torch.load(loadfile,map_location=torch.device('cpu'), weights_only=False))
                cnn.to(device)

                for igmt,gmt in enumerate(gmtvec):

                    obsinputgmt_t = gmt*torch.ones((len(obsinputgmt),1),dtype=torch.float32)
                    obsinputvectors_t = torch.cat((obsinputgmt_t,obsinputpriorrecord_t),axis=-1)

                    with torch.no_grad():
                        cnn.eval()
                        obspred = cnn(obsinput_t.to(device), obsinputvectors_t.to(device)).cpu().numpy()
                        obspred_climo = cnn(obsinput_meaninput_anom.to(device), obsinputvectors_t[[-1]].to(device)).cpu().numpy()

                    predsout[ilon,iseed,igmt,:] = obspred[:,1]
                    predsout_climo[ilon,iseed,igmt] = obspred_climo[:,1]
                    
    allperturbpred = [predsout,predsout_climo]
        
    with open(predsoutfile,'wb') as f:
        pickle.dump(allperturbpred,f)
# %%
