#!/usr/bin/env python3

from predictability_emergence import DataHolder
from predictability_emergence import buildmodel

import importlib as imp
import numpy as np
import torch

import glob
import json
import pickle
import argparse
import logging
#%%

imp.reload(DataHolder)

# reliability curves for the ENSEMBLE MEAN prediction at each grid point, fitted on
# held-out test members (no seed ever trained or validated on them), so the curve
# matches the ensemble that is applied to observations

def get_best_files(filelist,n_best):
    results = []
    for file in filelist:
        with open(file, 'r') as f:
            results.append(json.load(f))

    allaccs = np.asarray([results[i]['val_accuracy'] for i in range(len(filelist))])
    allnulls = np.asarray([results[i]['val_class_imbalance'] for i in range(len(filelist))])
    allseeds = np.asarray([results[i]['seed'] for i in range(len(filelist))])
    bestseedinds = np.argsort(allaccs-allnulls)[-n_best:]

    bestseeds = allseeds[bestseedinds]

    return bestseeds

def reliability(p, y, w=None, n_bins=20, strategy="uniform", edges=None):
    p = np.asarray(p, float); y = np.asarray(y, float)
    w = np.ones_like(p) if w is None else np.asarray(w, float)

    if edges is None:
        edges = (np.quantile(p, np.linspace(0, 1, n_bins + 1)) if strategy == "quantile"
                 else np.linspace(0, 1, n_bins + 1))
    n_bins = len(edges) - 1
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)

    pm = np.full(n_bins, np.nan)
    om = np.full(n_bins, np.nan)
    n  = np.zeros(n_bins)
    for b in range(n_bins):
        m = idx == b
        if w[m].sum() > 0:
            pm[b] = np.average(p[m], weights=w[m])
            om[b] = np.average(y[m], weights=w[m])
            n[b]  = w[m].sum()

    ok = n > 0
    ece = np.sum(n[ok] * np.abs(pm[ok] - om[ok])) / n[ok].sum()
    centres = 0.5 * (edges[:-1] + edges[1:])
    return pm, om, n, ece, centres

def main():
    logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    )

    # setting up parser
    parser = argparse.ArgumentParser(prog="eval_calib_ensemble")
    # main parameters
    parser.add_argument("--lat", type=int, default=0)
    parser.add_argument("--n_best", type=int, default=3) # 3 = best 3 seeds, 10 = all seeds
    parser.add_argument("--calib_members", nargs=2, type=int, default=[38, 44]) # fit curves on these, keep the rest of test for evaluation
    parser.add_argument("--n_bins", type=int, default=20)
    # just in case parameters
    parser.add_argument("--outputavgtime", type=int, default=5)
    parser.add_argument("--ssps", nargs="+", default=["126", "245", "370", "585"])
    # CHECK: baseline must match training (trainnn.py / nosst_trainnn.py default [1900,1950]),
    # otherwise records are defined differently to what the networks learned
    parser.add_argument("--experiment_era", nargs=2, type=int, default=[1980, 2100])
    parser.add_argument("--baseline_era", nargs=2, type=int, default=[1900, 1950])
    parser.add_argument("--input_length", type=int, default=10)
    parser.add_argument("--in_res", type=int, default=4)
    parser.add_argument("--out_res", type=int, default=10)
    parser.add_argument("--time_range", nargs=2, type=int, default=[1900, 2100])
    parser.add_argument("--file_front", type=str, default="MPI_")
    parser.add_argument("--model_file_front", type=str, default="MPI_recordtemp_")
    parser.add_argument("--ann_model_file_front", type=str, default="MPI_recordtemp_nosst")
    parser.add_argument("--input_var", type=str, default="tos")
    parser.add_argument("--output_var", type=str, default="tas")
    parser.add_argument("--n_train", type=int, default=25)
    parser.add_argument("--n_val", type=int, default=13)
    parser.add_argument("--data_dir", type=str, default="../data/")

    args = parser.parse_args()

    lat = args.lat
    n_best = args.n_best
    calib_members = np.arange(args.calib_members[0],args.calib_members[1])
    n_bins = args.n_bins
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
    annmodelfilefront = args.ann_model_file_front
    input_var = args.input_var
    output_var = args.output_var
    n_train = args.n_train
    n_val = args.n_val
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
    logging.info("Ensembling best %s seeds, calibrating on members %s", n_best, calib_members)

    # get the data

    AllData = DataHolder.MPIInputOutput_SSPlist(params,ssp_list)

    # train/val slots are unused here, calibration members go in the test slot
    trainvaltest = [np.arange(n_train),np.arange(n_train,n_train+n_val),calib_members]

    calibtag = "_best"+str(n_best)+"_calibmembers_"+str(calib_members[0])+"-"+str(calib_members[-1]+1)

    for ilon,lon in enumerate(AllData.output_lon):

        print(lon)

        metricsout = "../metrics/"+ model_file_front+"avgtime_"+str(outputavgtime)+"_allssps_lat_"+str(lat)+"_lon_"+str(lon)+"_seed_*.json"
        filelist = glob.glob(metricsout)

        calibout = "../metrics/"+model_file_front+"avgtime_"+str(outputavgtime)+"_allssps"+calibtag+"_lat_"+str(lat)+"_lon_"+str(lon)+"_reliability.pkl"

        if len(filelist)!=0:
            logging.info("Models exist, proceeding")

            _, _, allcalib = AllData.trainvaltest_recordmax_withrecordmax(trainvaltest,experiment_era,baseline_era,input_length,outputavgtime,lat,lon)

            inputcalib, inputcalibGMT, outputcalib = DataHolder.tensortime_onehot_withrecordmax(allcalib,nclasses=2)

            calibtrueclass = np.argmax(outputcalib.numpy(),axis=1)
            calibimbalance = float(np.mean(calibtrueclass))
            logging.info("Calibration base rate is %s", calibimbalance)

            # same seeds for CNN and ANN, as in ObsPredictMetrics
            bestseeds = get_best_files(filelist,n_best)
            print(bestseeds)

            allcalibpred = np.zeros((len(bestseeds),len(calibtrueclass)))+np.nan
            allcalibpred_ann = np.zeros((len(bestseeds),len(calibtrueclass)))+np.nan

            for iseed,seed in enumerate(bestseeds):
                # load the models

                loadfile = "../models/"+ model_file_front+ "avgtime_"+ str(outputavgtime)+ "_allssps_lat_"+ str(lat)+ "_lon_"+ str(lon)+ "_seed_"+ str(seed)+ ".pt"
                cnn = buildmodel.CNNclassifier(inputcalib, inputcalibGMT, 2).to('cpu')
                cnn.load_state_dict(torch.load(loadfile,map_location=torch.device('cpu'), weights_only=False))
                cnn.to(device)

                annloadfile = "../models/"+annmodelfilefront+"avgtime_"+str(outputavgtime)+"_allssps_lat_"+str(lat)+"_lon_"+str(lon)+"_seed_"+str(seed)+".pt"
                ann = buildmodel.ANNclassifier(inputcalibGMT, 2).to('cpu')
                ann.load_state_dict(torch.load(annloadfile,map_location=torch.device('cpu'), weights_only=False))
                ann.to(device)

                with torch.no_grad():
                    cnn.eval()
                    ann.eval()
                    calibpred = cnn(inputcalib.to(device), inputcalibGMT.to(device)).cpu().numpy()
                    calibpred_ann = ann(inputcalibGMT.to(device)).cpu().numpy()

                allcalibpred[iseed] = calibpred[:,1]
                allcalibpred_ann[iseed] = calibpred_ann[:,1]

            # calibrate the ensemble mean, which is what gets used on obs
            predmean = np.mean(allcalibpred,axis=0)
            predmean_ann = np.mean(allcalibpred_ann,axis=0)

            pm, om, n, ece, centres = reliability(predmean,calibtrueclass,n_bins=n_bins)
            pm_ann, om_ann, n_ann, ece_ann, _ = reliability(predmean_ann,calibtrueclass,n_bins=n_bins)

            logging.info("CNN ECE %s, ANN ECE %s", ece, ece_ann)

            # raw predictions saved too so curves can be refit (other bins, interpolation) without rerunning
            result = {
                "lat": lat,
                "lon": lon,
                "seeds": bestseeds,
                "calib_members": calib_members,
                "centres": centres,
                "predictedmass": pm,
                "observedmass": om,
                "counts": n,
                "ece": ece,
                "predictedmass_ann": pm_ann,
                "observedmass_ann": om_ann,
                "counts_ann": n_ann,
                "ece_ann": ece_ann,
                "calib_base_rate": calibimbalance,
                "ensemblepred": predmean,
                "ensemblepred_ann": predmean_ann,
                "trueclass": calibtrueclass,
            }

            with open(calibout,'wb') as f:
                pickle.dump(result,f)

# %%
if __name__ == "__main__":
    main()
