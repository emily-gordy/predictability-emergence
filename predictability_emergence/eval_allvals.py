#!/usr/bin/env python3

from predictability_emergence import DataHolder
from predictability_emergence import buildmodel

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
import argparse
import logging
#%% 

imp.reload(DataHolder)

def f1_threshold_grid(scores, y, thresholds):

    s = scores[None, :]               # shape (1, n)
    t = thresholds[:, None]           # shape (T, 1)
    pos = (y == 1)[None, :]           # shape (1, n)

    pred = s >= t                     # shape (T, n): row k = decisions at threshold k

    tp = np.sum(pred & pos, axis=1)   # predicted 1, actually 1
    fp = np.sum(pred & ~pos, axis=1)  # predicted 1, actually 0
    fn = np.sum(~pred & pos, axis=1)  # predicted 0, actually 1

    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)

    return f1

def posneg_threshold_grid(scores, y, thresholds):

    numneg = np.empty(len(thresholds))
    numpos = np.empty(len(thresholds))

    for it,t in enumerate(thresholds):
        predsel = scores >= t                   

        numneg[it] = 1-np.mean(y[~predsel])
        numpos[it] = np.mean(y[predsel])

    return [numneg,numpos]

def save_metrics(accuracy, class_imbalance, json_file):
    result = {
        "test_accuracy": accuracy.tolist(),
        "test_class_imbalance": class_imbalance
    }
    
    with open(json_file, 'w') as f:
        json.dump(result, f, indent=2)

def get_best_files(filelist,n_best):
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

def main():
    # setting up logging
    # log_filename = datetime.datetime.now().strftime("trainnn_%Y-%m-%d.log")
    logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    # filename=log_filename,
                    # filemode='w'
                    )
    # define a Handler which writes INFO messages or higher to the sys.stderr
    # console = logging.StreamHandler()
    # console.setLevel(logging.INFO)
    # # set a format which is simpler for console use
    # formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # # tell the handler to use this format
    # console.setFormatter(formatter)
    # # add the handler to the root logger
    # logging.getLogger().addHandler(console)

    # setting up parser
    parser = argparse.ArgumentParser(prog="trainnn")
    # main parameters
    parser.add_argument("--n_best", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    # just in case parameters
    parser.add_argument("--outputavgtime", type=int, default=5)
    parser.add_argument("--ssps", nargs="+", default=["126", "245", "370", "585"])
    # parser.add_argument("--experiment_era", nargs=2, type=int, default=[1950, 2100])
    # parser.add_argument("--baseline_era", nargs=2, type=int, default=[1900, 1950])
    parser.add_argument("--experiment_era", nargs=2, type=int, default=[1980, 2100])
    parser.add_argument("--baseline_era", nargs=2, type=int, default=[1950, 1980])
    parser.add_argument("--input_length", type=int, default=10)
    parser.add_argument("--in_res", type=int, default=4)
    parser.add_argument("--out_res", type=int, default=10)
    parser.add_argument("--time_range", nargs=2, type=int, default=[1900, 2100])
    parser.add_argument("--file_front", type=str, default="MPI_")
    parser.add_argument("--model_file_front", type=str, default="MPI_recordtemp_")
    parser.add_argument("--input_var", type=str, default="tos")
    parser.add_argument("--output_var", type=str, default="tas")
    parser.add_argument("--n_train", type=int, default=25)
    parser.add_argument("--n_val", type=int, default=13)
    parser.add_argument("--test", nargs=2, type=int, default=[38, 50])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.05)
    # parser.add_argument("--ridge_pen", type=float, default=1e-6) # is this used?
    parser.add_argument("--lr_patience", type=int, default=7)
    parser.add_argument("--early_stopping_patience", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--momentum", type=float, default=0.5)
    parser.add_argument("--data_dir", type=str, default="../data/")

    args = parser.parse_args()

    iseed = args.seed
    n_best = args.n_best
    outputavgtime = args.outputavgtime
    ssp_list = args.ssps
    experiment_era = args.experiment_era
    baseline_era = args.baseline_era
    input_length = args.input_length
    in_res = args.in_res
    out_res = args.out_res
    time_range = args.time_range
    file_front = args.file_front
    model_file_front = args.model_file_front
    input_var = args.input_var
    output_var = args.output_var
    n_train = args.n_train
    n_val = args.n_val
    test = np.arange(args.test[0],args.test[1])
    batch_size = args.batch_size
    lr = args.lr
    # ridge_pen = args.ridge_pen
    lr_patience = args.lr_patience
    early_stopping_patience = args.early_stopping_patience
    epochs = args.epochs
    momentum = args.momentum
    data_dir = args.data_dir

    annmodelfilefront = "MPI_recordtemp_nosst"
    # make parameter dictionary to be passed to DataHolder
    params = {
        "input_length": input_length,
        "outputavgtime": outputavgtime,
        "out_res": out_res,
        "timerange": time_range,
        "filefront": file_front,
        "inres": in_res,
        "inputvar": input_var,
        "outputvar": output_var,
        "data_dir": data_dir,
    }

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available() & torch.backends.mps.is_built():
        device = 'mps'
    else:
        device='cpu'

    logging.info("Using device: %s", device)

    # get the data

    AllData = DataHolder.MPIInputOutput_SSPlist(params,ssp_list)
    # landmask = np.isnan(AllData.alloutput[0][0,0])

    # split data
    
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
    
    dummylon = 0
    seed = seedlist[iseed]

    torch.manual_seed(seed)
    np.random.seed(seed)
    trainval = np.random.choice(n_train+n_val,n_train+n_val,replace=False)

    trainvaltest = [trainval[:n_train],trainval[n_train:n_train+n_val],test]

    thresholds = np.linspace(0.01,0.99,99)

    allthresval = np.empty((18,36,len(thresholds)))+np.nan
    allthresval_ann = np.empty((18,36,len(thresholds)))+np.nan

    all5050val = np.empty((18,36,2,len(thresholds)))+np.nan
    all5050val_ann = np.empty((18,36,2,len(thresholds)))+np.nan

    valmetricsout = "../metrics/"+model_file_front+"avgtime_"+str(outputavgtime)+"_ssp370_validation_decisionboundary_seed"+str(seed)+".pkl"

    for ilat,lat in enumerate(AllData.output_lat):

        for ilon,lon in enumerate(AllData.output_lon):

            print(lon)

            metricsout = "../metrics/"+ model_file_front+"avgtime_"+str(outputavgtime)+"_allssps_lat_"+str(lat)+"_lon_"+str(lon)+"_seed"+str(seed)+".json"
            filelist = glob.glob(metricsout)

            if len(filelist)!=0:
                logging.info("Models exist, proceeding")

                _, allval, _ = AllData.trainvaltest_recordmax_withrecordmax(trainvaltest,experiment_era,baseline_era,input_length,outputavgtime,lat,lon)
                
                inputval, inputvalGMT, outputval = DataHolder.tensortime_onehot_withrecordmax(allval,nclasses=2)

                valtrueclass = np.argmax(outputval.numpy(),axis=1)
                valimbalance = np.mean(outputval[:,1].numpy())
                valimbalance = [float((1-valimbalance)),float(valimbalance)]

                print("Test imbalance is %s:%s", valimbalance[0], valimbalance[1])

                    # load the model

                loadfile = "../models/"+ model_file_front+ "avgtime_"+ str(outputavgtime)+ "_allssps_lat_"+ str(lat)+ "_lon_"+ str(lon)+ "_seed_"+ str(seed)+ ".pt"
                cnn = buildmodel.CNNclassifier(inputval, inputvalGMT, 2).to('cpu')
                cnn.load_state_dict(torch.load(loadfile,map_location=torch.device('cpu'), weights_only=False))
                cnn.to(device)

                annloadfile = "../models/"+annmodelfilefront+"avgtime_"+str(outputavgtime)+"_allssps_lat_"+str(lat)+"_lon_"+str(lon)+"_seed_"+str(seed)+".pt"
                ann = buildmodel.ANNclassifier(inputvalGMT, 2).to('cpu')
                ann.load_state_dict(torch.load(annloadfile,map_location=torch.device('cpu'), weights_only=False))
                ann.to(device)

                with torch.no_grad():
                    cnn.eval()
                    valpred = cnn(inputval.to(device), inputvalGMT.to(device)).cpu().numpy()

                    ann.eval()
                    valpred_ann = ann(inputvalGMT.to(device)).cpu().numpy()

                allvalpred = valpred[:,1].squeeze()
                allvalpred_ann = valpred_ann[:,1].squeeze()

                allthresval[ilat,ilon] = f1_threshold_grid(allvalpred,valtrueclass,thresholds)
                allthresval_ann[ilat,ilon] = f1_threshold_grid(allvalpred_ann,valtrueclass,thresholds)

                all5050val[ilat,ilon] = posneg_threshold_grid(allvalpred,valtrueclass,thresholds)
                all5050val_ann[ilat,ilon] = posneg_threshold_grid(allvalpred_ann,valtrueclass,thresholds)


    with open(valmetricsout,'wb') as f:
        pickle.dump([allthresval,
                     allthresval_ann,
                     all5050val,
                     all5050val_ann,
                     thresholds],f)

    # with open(valpredfile_ann,'wb') as f:
    #     pickle.dump(allvalpred_ann,f)

    # with open(valtruefile,'wb') as f:
    #     pickle.dump(allvaltrue,f)

# %%
if __name__ == "__main__":
    main()