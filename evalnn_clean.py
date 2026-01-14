#%%

import DataHolder
import buildmodel

import importlib as imp
#import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import time

import glob
import sys
import json
import os
import pickle

#%% 

# iseed = int(sys.argv[1])
# iseed = 0

# num_workers = int(os.environ.get('DATALOADER_WORKERS', 4))
# print(f"Using {num_workers} DataLoader workers")

# latsel = 40
# lonsel = 240

ilatsel = int(sys.argv[1])

#%%

imp.reload(DataHolder)

# set user parameters
ssplist = ["126","245","370","585"]

experiment_era = [1950,2100]
baselineera = [1900,1950]

inputlength = 10
outputavgtime = 5

# data params
outres = 10
timerange = [1900,2100]
filefront = "MPI_"
modelfilefront = "MPI_recordtemp_"
inres = 4
inputvar = 'tos'
outputvar = 'tas'

ntrain = 25
nval = 13
test = np.arange(38,50)

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

# seed = seedlist[iseed]

# some training params

batch_size = 128
lr = 0.05
ridge_pen = 1e-6
lr_patience = 7
early_stopping_patience = 20
epochs = 2000

momentum = 0.5

# make parameter dictionary to be passed to DataHolder

params = {
    "inputlength": inputlength,
    "outputavgtime": outputavgtime,
    "outres": outres,
    "timerange": timerange,
    "filefront": filefront,
    "inres": inres,
    "inputvar":inputvar,
    "outputvar":outputvar,
    "seedlist": seedlist,
}

# get the data

AllData = DataHolder.MPIInputOutput_SSPlist(params,ssplist)
landmask = np.isnan(AllData.alloutput[0][0,0])

def save_metrics(accuracy, class_imbalance, json_file):
    result = {
        "test_accuracy": accuracy.tolist(),
        "test_class_imbalance": class_imbalance
    }
    
    with open(json_file, 'w') as f:
        json.dump(result, f, indent=2)

def get_best_files(filelist,n_best=3):
    results = []
    for file in filelist:
        with open(file, 'r') as f:
            results.append(json.load(f))
    
    allaccs = np.asarray([results[i]['val_accuracy'] for i in range(len(filelist))])
    allnulls = np.asarray([results[i]['val_class_imbalance'] for i in range(len(filelist))])
    allseeds = np.asarray([results[i]['seed'] for i in range(len(filelist))])
    bestseedinds = np.argsort(allaccs-allnulls)[-n_best:]

    # print(bestseedinds)

    bestseeds = allseeds[bestseedinds]
    # bestfiles = []
    # for bestind in bestseedinds:
    #     bestfile = filelist[bestind]
    #     bestfiles.append(bestfile)

    return bestseeds

def confacc(predclass,trueclass,predconf):

    predcorr = predclass==trueclass
    percentiles = np.arange(0,100,5)
    accper = np.empty(20)
    for iper,per in enumerate(percentiles):

        perboo = np.percentile(predconf,per)
        accper[iper] = np.mean(predcorr[predconf>perboo])

    return accper

# set device

if torch.cuda.is_available():
   device = 'cuda'
elif torch.backends.mps.is_available() & torch.backends.mps.is_built():
   device = 'mps'
else:
   device='cpu'

print(f"Using device: {device}")

# split data

torch.manual_seed(seedlist[0])
np.random.seed(seedlist[0])

lat = AllData.output_lat[ilatsel]
dummylon = 0
n_best = 3

trainval = np.random.choice(ntrain+nval,ntrain+nval,replace=False)

trainvaltest = [trainval[:ntrain],trainval[ntrain:ntrain+nval],test]

_, _, alltest = AllData.trainvaltest_recordmax(trainvaltest,experiment_era,baselineera,inputlength,outputavgtime,lat,dummylon)

inputtest, inputtestGMT, _ = DataHolder.tensortime_onehot(alltest,nclasses=2)

alltestpred = np.zeros((len(AllData.output_lon),n_best,len(inputtestGMT)))
alltesttrue = np.zeros((len(AllData.output_lon),len(inputtestGMT)))
testpredfile = "predictions/"+modelfilefront+"avgtime"+str(outputavgtime)+"_allssps_lat"+str(lat)+"_testing.pkl"
testtruefile = "predictions/"+modelfilefront+"avgtime"+str(outputavgtime)+"_allssps_lat"+str(lat)+"_truetesting.pkl"

for ilon,lon in enumerate(AllData.output_lon):

    metricsout = "metrics/"+modelfilefront+"avgtime"+str(outputavgtime)+"_allssps_lat"+str(lat)+"_lon"+str(lon)+"_seed*.json"
    filelist = glob.glob(metricsout)

    testmetricsout = "metrics/"+modelfilefront+"avgtime"+str(outputavgtime)+"_allssps_lat"+str(lat)+"_lon"+str(lon)+"_testing.json"

    if len(filelist)!=0:
        print('models exist, proceeding')

        bestseeds = get_best_files(filelist,n_best)
        # print(bestseeds)
        _, _, outputtestall = AllData.trainvaltest_recordmax_outputonly(trainvaltest,experiment_era,baselineera,inputlength,outputavgtime,lat,lon)

        inputtest,inputtestGMT,outputtest = DataHolder.tensortime_onehot_inputoutput(alltest,outputtestall,nclasses=2)
        testtrueclass = np.argmax(outputtest.numpy(),axis=1)
        alltesttrue[ilon] = testtrueclass

        testimbalance = np.mean(outputtestall)
        testimbalance = [(1-testimbalance),testimbalance]
        nullimbalance = np.max(np.asarray(testimbalance))
        print("test imbalance is "+ str(testimbalance[0]) + ":" + str(testimbalance[1]))

        for iseed,seed in enumerate(bestseeds):
            # load the model

            loadfile = "models/"+modelfilefront+"avgtime"+str(outputavgtime)+"_allssps_lat"+str(lat)+"_lon"+str(lon)+"_seed"+str(seed)+".pt"
            cnn = buildmodel.CNNclassifier(inputtest, inputtestGMT, 2).to('cpu')
            cnn.load_state_dict(torch.load(loadfile,map_location=torch.device('cpu'), weights_only=False))
            cnn.to(device)

            with torch.no_grad():
                cnn.eval()
                testpred = cnn(inputtest.to(device), inputtestGMT.to(device)).cpu().numpy()

            alltestpred[ilon,iseed,] = testpred[:,1]

        predmean = np.mean(alltestpred[ilon],axis=0) # confidence in positive class
        predclass = np.round(predmean) # prediction class
        predconf = np.where(predmean<0.5,1-predmean,predmean)

        accuracy = confacc(predclass,testtrueclass,predconf)

        save_metrics(accuracy, testimbalance, testmetricsout)

with open(testpredfile,'wb') as f:
    pickle.dump(alltestpred,f)

with open(testtruefile,'wb') as f:
    pickle.dump(alltesttrue,f)

# %%
