#%%

import DataHolder

import importlib as imp
#import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

iseed = 0

latsel = -30
lonsel = 140

#%%

imp.reload(DataHolder)

# set user parameters
ssplist = ["126","245","370","585"]

experiment_era = [1950,2100]
baselineera = [1900,1950]

inputlength = 10
outputavgtime = 10

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

seed = seedlist[iseed]

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

np.random.seed(seed)

trainval = np.random.choice(ntrain+nval,ntrain+nval,replace=False)

trainvaltest = [trainval[:ntrain],trainval[ntrain:ntrain+nval],test]

alltrain, allval, alltest = AllData.trainvaltest_recordmax(trainvaltest,experiment_era,baselineera,inputlength,outputavgtime,latsel,lonsel)

inputtrainGMT = alltrain[1]
outputtrain = alltrain[2]

inputvalGMT = allval[1]
outputval = allval[2]

#%%

# Initialize the model
model = LogisticRegression()

# Fit the model
model.fit(inputtrainGMT, outputtrain)

# Make predictions
y_pred = model.predict(inputvalGMT)
y_pred_proba = model.predict_proba(inputvalGMT)  # Get probabilities

# Evaluate
accuracy = accuracy_score(outputval, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(outputval, y_pred))

# Access coefficients
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")

# %%
