#!/usr/bin/env python3

from predictability_emergence import DataHolder
from predictability_emergence import commonConfig

import numpy as np
import torch
import argparse
import logging
import os

def main():
    # setting up logging
    logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    )

    # setting up parser
    parser = argparse.ArgumentParser(prog="basepred")

    # common, default parameters
    parser.add_argument("--config", default="config/common_params.json", help="Path to common JSON config")

    # the common config arguments can be overidden by command line arguments
    parser.add_argument("--outputavgtime", type=int, default=5)
    parser.add_argument("--ssps", nargs="+",
                        default=["126", "245", "370", "585"],
                        choices=["126", "245", "370", "585"], 
                        help="SSP scenarios")
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
    parser.add_argument("--test", nargs=2, type=int, default=[38, 50])
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--output_dir", type=str, default="predictions/")

    args = commonConfig.apply_common_config_and_parse_args(parser)

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
    data_dir = args.data_dir
    output_dir = args.output_dir

    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {output_dir}: {e}")

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
    
    dummylon = 0

    trainval = np.random.choice(n_train+n_val,n_train+n_val,replace=False)

    trainvaltest = [trainval[:n_train],trainval[n_train:n_train+n_val],test]

    for _, lat in enumerate(AllData.output_lat):

        _, _, alltest = AllData.trainvaltest_recordmax(trainvaltest,experiment_era,baseline_era,input_length,outputavgtime,lat,dummylon)

        inputtest, inputtestGMT, _ = DataHolder.tensortime_onehot(alltest,nclasses=2)

        alltesttrue = np.zeros((len(AllData.output_lon),len(inputtestGMT)))
        testtruefile = output_dir+model_file_front+"avgtime"+str(outputavgtime)+"_allssps_lat"+str(lat)+"_truetesting.csv"

        # add lat loop
        # can we vectorize this?

        for ilon,lon in enumerate(AllData.output_lon):
                _, _, outputtestall = AllData.trainvaltest_recordmax_outputonly(trainvaltest,experiment_era,baseline_era,input_length,outputavgtime,lat,lon)

                inputtest,inputtestGMT,outputtest = DataHolder.tensortime_onehot_inputoutput(alltest,outputtestall,nclasses=2)
                testtrueclass = np.argmax(outputtest.numpy(),axis=1)
                alltesttrue[ilon] = testtrueclass

        np.savetxt(testtruefile, alltesttrue, delimiter = ',')

if __name__ == "__main__":
    main()