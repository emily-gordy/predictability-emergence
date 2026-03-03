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
    parser = argparse.ArgumentParser(prog="evalnn")
    # main parameters
    parser.add_argument("--lat", type=int, default=0)
    # just in case parameters
    parser.add_argument("--outputavgtime", type=int, default=5)
    parser.add_argument("--ssps", nargs="+", default=["126", "245", "370", "585"])
    parser.add_argument("--experiment_era", nargs=2, type=int, default=[1950, 2100])
    parser.add_argument("--baseline_era", nargs=2, type=int, default=[1900, 1950])
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
    parser.add_argument("--test", nargs=2, type=int, default=[38, 49])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--data_dir", type=str, default="../data/")

    args = parser.parse_args()

    lat = args.lat
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
    test = np.array(args.test)
    batch_size = args.batch_size
    data_dir = args.data_dir

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
    landmask = np.isnan(AllData.alloutput[0][0,0])

    # split data

    dummylon = 0

    trainval = np.random.choice(n_train+n_val,n_train+n_val,replace=False)

    trainvaltest = [trainval[:n_train],trainval[n_train:n_train+n_val],test]

    _, _, alltest = AllData.trainvaltest_recordmax(trainvaltest,experiment_era,baseline_era,input_length,outputavgtime,lat,dummylon)

    inputtest, inputtestGMT, _ = DataHolder.tensortime_onehot(alltest,nclasses=2)

    alltestpred = np.zeros((len(AllData.output_lon),n_best,len(inputtestGMT)))
    alltesttrue = np.zeros((len(AllData.output_lon),len(inputtestGMT)))
    testpredfile = "predictions/"+model_file_front+"avgtime"+str(outputavgtime)+"_allssps_lat"+str(lat)+"_testing.pkl"
    testtruefile = "predictions/"+model_file_front+"avgtime"+str(outputavgtime)+"_allssps_lat"+str(lat)+"_truetesting.pkl"

    for ilon,lon in enumerate(AllData.output_lon):

        metricsout = model_file_front+"avgtime"+str(outputavgtime)+"_allssps_lat"+str(lat)+"_lon"+str(lon)+"_seed*.json"
        filelist = glob.glob(metricsout)

        testmetricsout = model_file_front+"avgtime"+str(outputavgtime)+"_allssps_lat"+str(lat)+"_lon"+str(lon)+"_testing.json"

        if len(filelist)!=0:
            logging.info("Models exist, proceeding")

            bestseeds = get_best_files(filelist,n_best)
            # print(bestseeds)
            _, _, outputtestall = AllData.trainvaltest_recordmax_outputonly(trainvaltest,experiment_era,baseline_era,input_length,outputavgtime,lat,lon)

            inputtest,inputtestGMT,outputtest = DataHolder.tensortime_onehot_inputoutput(alltest,outputtestall,nclasses=2)

            testimbalance = np.mean(outputtestall)
            testimbalance = [(1-testimbalance),testimbalance]
            nullimbalance = np.max(np.asarray(testimbalance))
            logging.info("Test imbalance is %s:%s", testimbalance[0], testimbalance[1])

            cnn = buildmodel.CNNclassifier(inputtest, inputtestGMT, 2).to('cpu')
            cnn.load_state_dict(torch.load(model_file,map_location=torch.device('cpu'), weights_only=False))
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

# %%
if __name__ == "__main__":
    main()