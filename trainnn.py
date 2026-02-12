#!/usr/bin/env python3

import DataHolder
import buildmodel

import importlib as imp

# import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import time

import pickle
import glob
import sys
import json
import os
import argparse
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s %(message)s", datefmt="%Y/%m/%d %H:%M:%S"
)

# training/validation functions


def train_loop(dataloader, cnn, loss_fn, optimizer, device):

    size = len(dataloader.dataset)

    cnn.train()
    for batch, (x1, x2, y) in enumerate(dataloader):
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)

        # Compute prediction and loss
        pred = cnn(x1, x2)
        # pred = model(X1)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(x1)
            logging.info("loss: {%s:>7f}  [{%s:>5d}/{%s:>5d}]", loss, current, size)


def val_loop(dataloader, model, loss_fn, optimizer, scheduler, device):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()

    all_pred = []
    all_true = []
    all_losses = []

    with torch.no_grad():
        for x1, x2, y in dataloader:
            # Move the data and targets to the specified device
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)

            # Forward pass
            pred = model(x1, x2)

            all_pred.extend(pred.detach().cpu().numpy())
            all_true.extend(y.detach().cpu().numpy())

            # Calculate individual losses
            batch_losses = loss_fn(
                pred, y
            )  # This returns a tensor of losses for each item in the batch
            all_losses.append(batch_losses.item())  # Convert tensor to scalar and store
    # Compute the overall loss (sum of individual losses divided by the number of samples)
    valid_loss = batch_losses.mean()

    all_pred = np.asarray(all_pred)[:, 1]
    all_true = np.asarray(all_true)

    logging.info("validation loss: {%s:>7f}", valid_loss)

    scheduler.step(valid_loss)
    after_lr = optimizer.param_groups[0]["lr"]
    logging.info("learning rate: {%s:>7f}", after_lr)

    return valid_loss


def model_checkpoint(
    model, val_loss, best_val_loss, epochs_no_improve, fileout, patience
):

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        logging.info("Loss improved, saving model to %s", fileout)
        torch.save(model.state_dict(), fileout)
        epochs_no_improve = 0
        earlystopping = 0
    else:
        epochs_no_improve += 1
        logging.info("Loss did not improve.")
        if epochs_no_improve == patience:
            earlystopping = 1
        else:
            earlystopping = 0

    return best_val_loss, earlystopping, epochs_no_improve


def save_metrics(seed, val_loss, accuracy, class_imbalance, json_file):
    result = {
        "seed": seed,
        "val_loss": val_loss,
        "val_accuracy": accuracy,
        "val_class_imbalance": class_imbalance,
    }

    with open(json_file, "w") as f:
        json.dump(result, f, indent=2)


def lightclassweighting(positiveclassimbalance, device):

    classimbalance = np.asarray([(1 - positiveclassimbalance), positiveclassimbalance])
    weights = np.max(classimbalance) / classimbalance

    classratio = np.max(classimbalance) / np.min(classimbalance)
    downweight = (
        0.85 * (classratio - 1) / classratio
    )  # amount to de-emphasize the class imbalance

    if classimbalance[0] < 0.47:
        weights_corrected = [weights[0] - downweight * weights[0], weights[1]]
        weights_corrected = torch.tensor(weights_corrected, dtype=torch.float32).to(
            device
        )

    elif classimbalance[0] > 0.53:
        weights_corrected = [weights[0], weights[1] - downweight * weights[1]]
        weights_corrected = torch.tensor(weights_corrected, dtype=torch.float32).to(
            device
        )

    else:
        weights_corrected = torch.tensor([1, 1], dtype=torch.float32).to(device)

    weights_corrected = weights_corrected / torch.sum(weights_corrected)

    return weights_corrected

def main():
    parser = argparse.ArgumentParser(prog="trainnn")
    # main parameters
    parser.add_argument("--lat", type=int, default=0)
    parser.add_argument("--lon", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
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

    args = parser.parse_args()

    lat = args.lat
    lon = args.lon
    iseed = args.seed
    outputavgtime = args.outputavgtime
    ssplist = args.ssps
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

    # # grab grid point
    # landmaskfile = "landmask" + str(out_res) + "x" + str(out_res) + ".pkl"

    # with open(landmaskfile, "rb") as f:
    #     land = pickle.load(f)

    ntrain = 25
    nval = 13
    test = np.arange(38, 50)

    seedlist = [
        62469869,
        71856281,
        47621498,
        10431957,
        50561320,
        72166634,
        18469465,
        92895735,
        57693846,
        22284750,
    ]

    # some training params

    batch_size = 128
    lr = 0.05
    ridge_pen = 1e-6
    lr_patience = 7
    early_stopping_patience = 20
    epochs = 200

    momentum = 0.5

    # make parameter dictionary to be passed to DataHolder

    params = {
        "input_length": input_length,
        "outputavgtime": outputavgtime,
        "out_res": out_res,
        "time_range": time_range,
        "file_front": file_front,
        "in_res": in_res,
        "input_var": input_var,
        "output_var": output_var,
        "seedlist": seedlist,
    }

    # get the data

    AllData = DataHolder.MPIInputOutput_SSPlist(params, ssplist)
    landmask = np.isnan(AllData.alloutput[0][0, 0])

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available() & torch.backends.mps.is_built():
        device = "mps"
    else:
        device = "cpu"
    logging.info("Using device: %s", device)

    # split data

    # for iseed,seed in enumerate(seedlist):

    seed = seedlist[iseed]

    torch.manual_seed(seed)
    np.random.seed(seed)

    trainval = np.random.choice(ntrain + nval, ntrain + nval, replace=False)

    trainvaltest = [trainval[:ntrain], trainval[ntrain : ntrain + nval], test]
    alltrain, allval, _ = AllData.trainvaltest_recordmax(
        trainvaltest, experiment_era, baseline_era, input_length, outputavgtime, lat, lon
    )

    inputtrain, inputtrainGMT, outputtrain = DataHolder.tensortime_onehot(
        alltrain, nclasses=2
    )
    inputval, inputvalGMT, outputval = DataHolder.tensortime_onehot(allval, nclasses=2)

    traindataset = TensorDataset(inputtrain, inputtrainGMT, outputtrain)
    train_loader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)

    valdataset = TensorDataset(inputval, inputvalGMT, outputval)
    val_loader = DataLoader(
        valdataset, batch_size=inputval.size(0), shuffle=False
    )  # all val in one batch maybe bad idea

    fileout = (
        model_file_front
        + "avgtime"
        + str(outputavgtime)
        + "_allssps_lat"
        + str(lat)
        + "_lon"
        + str(lon)
        + "_seed"
        + str(seed)
        + ".pt"
    )
    metricsout = (
        model_file_front
        + "avgtime"
        + str(outputavgtime)
        + "_allssps_lat"
        + str(lat)
        + "_lon"
        + str(lon)
        + "_seed"
        + str(seed)
        + ".json"
    )

    # filecheck = glob.glob(metricsout)
    # if len(filecheck)==0:

    outputtrainall, outputvalall, _ = AllData.trainvaltest_recordmax_outputonly(
        trainvaltest, experiment_era, baseline_era, input_length, outputavgtime, lat, lon
    )

    inputtrain, inputtrainGMT, outputtrain = DataHolder.tensortime_onehot_inputoutput(
        alltrain, outputtrainall, nclasses=2
    )
    inputval, inputvalGMT, outputval = DataHolder.tensortime_onehot_inputoutput(
        allval, outputvalall, nclasses=2
    )

    traindataset = TensorDataset(inputtrain, inputtrainGMT, outputtrain)
    train_loader = DataLoader(
        traindataset,
        batch_size=batch_size,
        shuffle=True,
    )  # num_workers=num_workers)

    valdataset = TensorDataset(inputval, inputvalGMT, outputval)
    val_loader = DataLoader(
        valdataset, batch_size=inputval.size(0), shuffle=False
    )  # all val in one batch maybe bad idea

    # lightly weight the class imbalance
    classimbalance = np.mean(outputtrainall)
    if classimbalance > 0:  # cant train where it never happens
        weights_corrected = lightclassweighting(classimbalance, device)

        print(weights_corrected)

        loss_fn = nn.CrossEntropyLoss(weight=weights_corrected)

        valimbalance = np.mean(outputvalall)
        valimbalance = [(1 - valimbalance), valimbalance]
        logging.info(
            "val imbalance is {%s:>7f}:{%s:>7f}", valimbalance[0], valimbalance[1]
        )

        # train the model

        cnn = buildmodel.CNNclassifier(
            inputtrain, inputtrainGMT, len(weights_corrected)
        ).to(device)

        optimizer = optim.SGD(
            cnn.parameters(),
            lr=lr,
            momentum=momentum,
        )
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            threshold=1e-4,
            factor=0.1,
            patience=lr_patience,
            cooldown=lr_patience,
            min_lr=1e-5,
        )

        loss = []

        best_val_loss = np.inf
        epochs_no_improve = 0
        logging.info("Starting training loop for seed %d", seed)
        for t in range(epochs):
            logging.info("Epoch %d\n-------------------------------", t + 1)
            time1 = time.time()

            train_loop(train_loader, cnn, loss_fn, optimizer, device)
            valid_loss = val_loop(
                val_loader, cnn, loss_fn, optimizer, scheduler, device
            )

            loss.append(valid_loss)
            time2 = time.time()
            logging.info("%s seconds per epoch", time2 - time1)
            best_val_loss, earlystopping, epochs_no_improve = model_checkpoint(
                cnn,
                valid_loss,
                best_val_loss,
                epochs_no_improve,
                fileout,
                early_stopping_patience,
            )
            if earlystopping == 1:
                logging.info("Early stopping after %d epochs.", t + 1)
                break

        cnn.load_state_dict(torch.load(fileout, weights_only=True))

        with torch.no_grad():
            cnn.eval()
            valpred = cnn(inputval.to(device), inputvalGMT.to(device))

        valpred = valpred.cpu().numpy()

        valpredclass = np.argmax(valpred, axis=1)
        valtrueclass = np.argmax(outputval.numpy(), axis=1)

        valacc = np.mean(valpredclass == valtrueclass)
        logging.info(
            "Best validation accuracy: %f on a bg acc of %f:%f",
            valacc,
            valimbalance[0],
            valimbalance[1],
        )

        val_randomchance = np.max(valimbalance)

        save_metrics(
            seed,
            best_val_loss.cpu().numpy().item(),
            valacc,
            val_randomchance,
            metricsout,
        )


# %%
if __name__ == "__main__":
    main()
