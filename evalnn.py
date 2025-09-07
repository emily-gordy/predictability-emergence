#%%
import DataHolder
import buildmodel

import importlib as imp
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import time

import sys

import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs

#%%

ssplist = ["126","245","370"]

# set user parameters

latsel = 40
lonsel = 250

historical_era = [1960,2000]
# tpercentile = 80
trainvaltest = [np.arange(25),np.arange(25,38),np.arange(38,50)]

inputlength = 10
outputavgtime = 3
outres = 10
# ssp = "126" 
timerange = [1900,2080]
filefront = "MPI_regridded_"+str(outputavgtime)+"yearavg_"
inres = 4
inputvar = 'tos'
outputvar = 'tas'

seedlist = [62469869,
            71856281,
            47621498,]

# ssp = "126"
#%%
plotnum=0

plt.figure(figsize=(12,8))

for ssp in ssplist:
    for tpercentile in [90,95]:

        params = {
            "inputlength": inputlength,
            "outputavgtime": outputavgtime,
            "outres": outres,
            "ssp": ssp,
            "timerange": timerange,
            "filefront": filefront,
            "inres": inres,
            "inputvar":inputvar,
            "outputvar":outputvar,
            "seedlist": seedlist,
        }

        #get the data

        AllData = DataHolder.MPIInputOutput(params)

        alltrain, allval, alltest = AllData.trainvaltest_binaryclassifier(trainvaltest, historical_era, tpercentile, latsel, lonsel)

        inputtrain, inputtrainGMT, outputtrain = DataHolder.tensortime_onehot(alltrain,nclasses=2)
        inputval, inputvalGMT, outputval = DataHolder.tensortime_onehot(allval,nclasses=2)
        inputtest, inputtestGMT, outputtest = DataHolder.tensortime_onehot(alltest,nclasses=2)

        
        outputval = torch.argmax(outputtest, dim=1).numpy()

        #baseline of beating gmt, need to get gmt

        gmtallmembers = AllData.alloutput
        avgeffectofGMT = np.mean(gmtallmembers[:25], axis=0)
        avgeffectofGMTabs = np.round(avgeffectofGMT, decimals=0)
        avgeffectofGMTval = np.tile(avgeffectofGMTabs, (len(trainvaltest[2])))

        allpred = []

        for iseed, seed in enumerate(seedlist):

            cnn = buildmodel.CNNclassifier(inputtrain, inputtrainGMT, outputtrain)
            fileout = "models/"+filefront+"ssp"+ssp+"_per"+str(tpercentile)+"_lat"+str(latsel)+"_lon"+str(lonsel)+"_seed"+str(seed)+".pt"
            cnn.load_state_dict(torch.load(fileout, weights_only=True))
            cnn.eval()

            with torch.no_grad():
                predval = cnn(inputtest, inputtestGMT)

            predvalclass = torch.argmax(predval, dim=1).numpy()
            allpred.append(predval.numpy())
            acc = np.mean(predvalclass == outputval)
            print("Seed ", str(seed), " val accuracy: ", str(acc))

        gmtonlyacc = np.mean(avgeffectofGMTval == outputval)
        print("GMT only val accuracy: ", str(gmtonlyacc))


        yearvec = np.arange(timerange[0]+inputlength, timerange[1]-outputavgtime+2)

        gmtplot = np.reshape(inputtrainGMT.numpy().squeeze(),(len(trainvaltest[0]),len(yearvec)))
        gmtplot = np.mean(gmtplot,axis=0)


        plt.subplot(3,2,plotnum+1)
        plt.plot(yearvec,avgeffectofGMT,color='xkcd:blue',label='GMT only',linewidth=1.3)
        plt.plot(yearvec,avgeffectofGMTabs,color='xkcd:brown',label='GMT only',linewidth=1.3)
        # plt.plot(yearvec,gmtplot)

        for iseed, seed in enumerate(seedlist):
            predseed = allpred[iseed]
            predseed = np.argmax(predseed, axis=1)
            predseed = np.reshape(predseed, (len(trainvaltest[2]), len(yearvec)))

            predseedmean = np.mean(predseed, axis=0)
            plt.plot(yearvec,predseedmean,label='Seed '+str(seed),linewidth=1.8,color='xkcd:indigo')

            for i in range(len(trainvaltest[2])):
                plt.scatter(yearvec, allpred[iseed][i*len(yearvec):(i+1)*len(yearvec),1], color='xkcd:light purple', s=8)

        plt.title('ssp '+ssp + ' tpercentile' + str(tpercentile))

        plotnum+=1 
plt.tight_layout()
plt.show()

# %%

# %%
