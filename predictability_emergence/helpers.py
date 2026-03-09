"""A wee module for useful lil functions"""

import numpy as np
import json

def firstpositive(pred,gmtvec):
    if np.sum(pred[pred>0.5])>1:
        firstpos = np.min(gmtvec[pred>0.5])
    else:
        firstpos = np.nan
    return firstpos

def firstconfident(pred,gmtvec):

    conf = np.where(pred<0.5,1-pred,pred)
    percentile = np.percentile(conf,50)
    if np.sum(1*((pred>0.5) & (conf>percentile)))>0:
        firstconf = np.min(gmtvec[(pred>0.5) & (conf>percentile)])
    else:
        firstconf = np.nan
    return firstconf

def firstcorrect(pred,true,gmtvec):
    predclass = np.round(pred)
    if np.sum((predclass==1) & (true==1))>0:
        firstcorrect = np.min(gmtvec[(predclass==1) & (true==1)])
    else:
        firstcorrect = np.nan
    return firstcorrect

def confacc(predclass,trueclass,predconf):

    predcorr = predclass==trueclass
    percentiles = np.arange(0,100,5)
    accper = np.empty(20)
    for iper,per in enumerate(percentiles):

        perboo = np.percentile(predconf,per)
        accper[iper] = np.mean(predcorr[predconf>perboo])

    return accper

def grabssp(data,issp,lenhist,lenfuture,nmems):

    if issp == 0:
        dataout = data[:nmems*lenhist,]
    else:
        dataout = data[nmems*(lenhist+(issp-1)*lenfuture):nmems*(lenhist+(issp)*lenfuture),]
    
    return dataout

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

    return bestseeds

def brierscore(preds,true):

    bs = np.mean((preds-true)**2)

    return bs