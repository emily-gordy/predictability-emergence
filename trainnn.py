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

import glob
import sys

#%%

iseed = int(sys.argv[1])

#%% set user parameters

latsel = 40
lonsel = 250
ssp = "245" 

historical_era = [1960,2000]
tpercentile = 90

inputlength = 10
outputavgtime = 3

# data params
outres = 10
timerange = [1950,2080]
filefront = "MPI_"
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
            634,
            9465,
            95735,
            93846,
            84750]

seed = seedlist[iseed]

#%% some training params

batch_size = 64
lr = 0.001
ridge_pen = 1e-6
lr_patience = 7
early_stopping_patience = 20
epochs = 2000

# make parameter dictionary to be passed to DataHolder

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

# get the data

AllData = DataHolder.MPIInputOutput(params)

# trainvaltest = [np.arange(25),np.arange(25,38),np.arange(38,50)]

trainval = np.random.choice(ntrain+nval,ntrain+nval,replace=False)

trainvaltest = [trainval[:ntrain],trainval[ntrain:ntrain+nval],test]

alltrain, allval, alltest = AllData.trainvaltest_binaryclassifier(trainvaltest, historical_era, inputlength, outputavgtime, tpercentile, latsel, lonsel)

inputtrain, inputtrainGMT, outputtrain = DataHolder.tensortime_onehot(alltrain,nclasses=2)
inputval, inputvalGMT, outputval = DataHolder.tensortime_onehot(allval,nclasses=2)



def train_loop(dataloader, cnn, loss_fn, optimizer,device):
    
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    cnn.train()
    for batch, (x1,x2,y) in enumerate(dataloader):

        x1, x2, y = x1.to(device), x2.to(device), y.to(device)

        # Compute prediction and loss
        pred = cnn(x1,x2)
        # pred = model(X1)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(x1)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def val_loop(dataloader, model, loss_fn, optimizer, scheduler, device):
        # Set the model to evaluation mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)

        all_pred = []
        all_true = []
        all_losses = []

        with torch.no_grad():

            for x1,x2, y in dataloader:
                # Move the data and targets to the specified device
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                
                # Forward pass
                pred = model(x1,x2)

                all_pred.extend(pred.detach().cpu().numpy())
                all_true.extend(y.detach().cpu().numpy())

                # Calculate individual losses
                batch_losses = loss_fn(pred, y)  # This returns a tensor of losses for each item in the batch
                all_losses.append(batch_losses.item())  # Convert tensor to scalar and store
        # Compute the overall loss (sum of individual losses divided by the number of samples)
        valid_loss = batch_losses.mean()

        all_pred = np.asarray(all_pred)[:,1]
        all_true = np.asarray(all_true)[:,1]

        print(f"validation loss: {valid_loss:>7f}")
        
        scheduler.step(valid_loss)
        after_lr = optimizer.param_groups[0]["lr"]
        print(f"learning rate: {after_lr:>7f}")

        return valid_loss

def model_checkpoint(model,val_loss,best_val_loss,epochs_no_improve,fileout,patience):

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        print("Loss improved, saving model to "+ fileout)
        torch.save(model.state_dict(), fileout)
        epochs_no_improve = 0
        earlystopping=0
    else:
        epochs_no_improve += 1
        print("Loss did not improve")
        if epochs_no_improve == patience:
            earlystopping=1
        else:
            earlystopping=0
    
    return best_val_loss, earlystopping, epochs_no_improve

#%% set device

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available() & torch.backends.mps.is_built():
    device = 'mps'
else:
    device='cpu'

print(f"Using device: {device}")

traindataset = TensorDataset(inputtrain,inputtrainGMT,outputtrain)
train_loader = DataLoader(traindataset,batch_size=batch_size,shuffle=True)

valdataset = TensorDataset(inputval,inputvalGMT,outputval)
val_loader = DataLoader(valdataset,batch_size=inputval.size(0),shuffle=False) # all val in one batch maybe bad idea


# lightly weight the class imbalance

classimbalance = outputtrain.mean(axis=0)

weights = max(classimbalance)/classimbalance

if classimbalance[0]<0.45:

    # weights = max(classimbalance)/classimbalance
    weights_corrected = [weights[0]-0.1,weights[1]]
    weights_corrected = torch.tensor(weights_corrected).to(device)

elif classimbalance[0]>0.55:

    # weights = max(classimbalance)/classimbalance
    weights_corrected = [weights[0],weights[1]-0.1]
    weights_corrected = torch.tensor(weights_corrected).to(device)

else:
    weights_corrected = torch.tensor([1,1]).to(device)

loss_fn = nn.CrossEntropyLoss(weight=weights_corrected)

valimbalance = outputval.mean(axis=0)
print("val imbalance is "+ str(valimbalance[0]) + ":" + str(valimbalance[1]))

# train the model

# for iseed, seed in enumerate(seedlist):

cnn = buildmodel.CNNclassifier(inputtrain, inputtrainGMT, outputtrain).to(device)

optimizer = optim.SGD(cnn.parameters(), 
                lr=lr,
                weight_decay=ridge_pen
                )
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, threshold=1e-4, factor=0.1, patience=lr_patience, cooldown=lr_patience, min_lr=5e-6)


torch.manual_seed(seed)
np.random.seed(seed)

loss = []

best_val_loss = np.inf
epochs_no_improve = 0

fileout = "models/"+filefront+"avgtime"+str(outputavgtime)+"_ssp"+ssp+"_per"+str(tpercentile)+"_lat"+str(latsel)+"_lon"+str(lonsel)+"_seed"+str(seed)+".pt"

print(f"class imbalance is {classimbalance[0]} to {classimbalance[1]}")

filecheck = glob.glob(fileout)
filecheck = []
if len(filecheck)==0:

    print('train loop')
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        time1 = time.time()

        train_loop(train_loader, cnn, loss_fn, optimizer, device)
        valid_loss = val_loop(val_loader, cnn, loss_fn, optimizer, scheduler, device)

        loss.append(valid_loss)
        time2 = time.time()
        print(f"{time2-time1:4f} seconds per epoch")
        best_val_loss, earlystopping, epochs_no_improve = model_checkpoint(cnn,valid_loss,best_val_loss,epochs_no_improve,fileout,early_stopping_patience)
        if earlystopping==1:

            print(f'Early stopping after {t+1} epochs.')
            break
    


    cnn.load_state_dict(torch.load(fileout, weights_only=True))

    with torch.no_grad():
        cnn.eval()
        valpred = cnn(inputval.to(device), inputvalGMT.to(device))

    valpred = valpred.detach().cpu().numpy()

    valpredclass = np.sum(valpred,axis=1)
    valtrueclass = np.sum(outputval.numpy(),axis=1)

    valacc = np.mean(valpredclass==valtrueclass)
    print("best accuracy = "+ str(valacc))
    print("on a bg acc of "+ str(valimbalance[0]) + ":" + str(valimbalance[1]))
