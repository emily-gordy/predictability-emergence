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

#%% 

iseed = int(sys.argv[1])
# iseed = 1

# preferably these would be loop variables i.e.

# for ilatsel,latsel in enumerate(latvec):
#    for ilonsel,lonsel in enumerate(lonvec):

latsel = 40
lonsel = 10

#%%

imp.reload(DataHolder)

# set user parameters
ssplist = ["126","245","370","585"]

experiment_era = [1950,2100]
baselineera = [1900,1950]

inputlength = 10
outputavgtime = 3
tpercentile = 80

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

# some training params

batch_size = 128
lr = 0.01
ridge_pen = 1e-6
lr_patience = 7
early_stopping_patience = 20
epochs = 2000

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

def GMTexprob(GMTvec,outputprobs):
    bins = np.arange(-1,5.05,0.05)

    GMTvec = np.squeeze(GMTvec)
    binprobs = []
    for ibin,binval in enumerate(bins[:-1]):
        if len(outputprobs.shape)==2:
            exs = outputprobs[(GMTvec>=binval) & (GMTvec<bins[ibin+1]),1]
        elif len(outputprobs.shape)==1:
            exs = outputprobs[(GMTvec>=binval) & (GMTvec<bins[ibin+1])]
        if len(exs) == 0:
            prob = np.nan
        else:
            prob = exs.mean()
        binprobs.append(prob)
    
    return np.asarray(binprobs),bins


def train_loop(dataloader, cnn, loss_fn, optimizer,device):
    
    size = len(dataloader.dataset)

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

class FocalLoss(nn.Module):
    def __init__(self, weights=[1,1], gamma=2, reduction='mean'):
        """
        Focal Loss implementation for multi-class classification
        
        Args:
            alpha (float): Weighting factor for rare class (default: 1)
            gamma (float): Focusing parameter (default: 2)
            reduction (str): Specifies the reduction to apply to the output:
                           'none' | 'mean' | 'sum' (default: 'mean')
        """
        super(FocalLoss, self).__init__()
        self.weights = weights
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: A float tensor of shape [batch_size, num_classes]
                   (raw logits from model)
            targets: A long tensor of shape [batch_size]
                    (ground truth class indices)
        """
        # Compute cross entropy
        # ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        focal_loss_noreduce = -1* self.weights * (1-inputs)**self.gamma * targets * torch.log(inputs) 
        focal_loss = focal_loss_noreduce.mean()
        # # Compute probabilities
        # pt = torch.exp(-ce_loss)
        
        # # Compute focal loss
        # focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # if self.reduction == 'mean':
        #     return focal_loss.mean()
        # elif self.reduction == 'sum':
        #     return focal_loss.sum()
        # else:
        return focal_loss

# set device

#if torch.cuda.is_available():
#    device = 'cuda'
#elif torch.backends.mps.is_available() & torch.backends.mps.is_built():
#    device = 'mps'
#else:
#    device='cpu'

device = 'cpu'

print(f"Using device: {device}")

# for ilatsel,latsel in enumerate(latvec):
#   for ilonsel,lonsel in enumerare(lonvec):

torch.manual_seed(seed)
np.random.seed(seed)

trainval = np.random.choice(ntrain+nval,ntrain+nval,replace=False)

trainvaltest = [trainval[:ntrain],trainval[ntrain:ntrain+nval],test]

alltrain, allval, alltest = AllData.trainvaltest_recordmax(trainvaltest,experiment_era,baselineera,inputlength,outputavgtime,latsel,lonsel)
# trainvaltest,historicalera,inputlength,outputlength,tpercentile,latsel,lonsel

inputtrain, inputtrainGMT, outputtrain = DataHolder.tensortime_classindex(alltrain,nclasses=2)
inputval, inputvalGMT, outputval = DataHolder.tensortime_classindex(allval,nclasses=2)

traindataset = TensorDataset(inputtrain,inputtrainGMT,outputtrain)
train_loader = DataLoader(traindataset,batch_size=batch_size,shuffle=True)

valdataset = TensorDataset(inputval,inputvalGMT,outputval)
val_loader = DataLoader(valdataset,batch_size=inputval.size(0),shuffle=False) # all val in one batch maybe bad idea

binprobs,bins = GMTexprob(inputtrainGMT.numpy(),outputtrain.numpy())

#%%
imp.reload(buildmodel)
# lightly weight the class imbalance

classimbalance = outputtrain.mean()
weights = max(classimbalance)/classimbalance

if classimbalance[0]<0.47:

    # weights = max(classimbalance)/classimbalance
    weights_corrected = [weights[0]-0.5*weights[0],weights[1]]
    weights_corrected = torch.tensor(weights_corrected).to(device)

elif classimbalance[0]>0.53:

    # weights = max(classimbalance)/classimbalance
    weights_corrected = [weights[0],weights[1]-0.5*weights[1]]
    weights_corrected = torch.tensor(weights_corrected).to(device)

else:
    weights_corrected = torch.tensor([1,1]).to(device)

weights_corrected = weights_corrected/torch.max(weights_corrected)
# weights_corrected = torch.tensor([0.8,1]).to(device)
print(weights_corrected)

loss_fn = nn.CrossEntropyLoss(weight=weights_corrected)

# loss_fn = FocalLoss(weights=weights_corrected,gamma=1.5)
valimbalance = outputval.mean(axis=0)
print("val imbalance is "+ str(valimbalance[0].numpy()) + ":" + str(valimbalance[1].numpy()))

# train the model

cnn = buildmodel.CNNclassifier(inputtrain, inputtrainGMT, outputtrain).to(device)

optimizer = optim.SGD(cnn.parameters(), 
                lr=lr,
                weight_decay=ridge_pen
                )
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, threshold=1e-4, factor=0.1, patience=lr_patience, cooldown=lr_patience, min_lr=5e-6)

loss = []

best_val_loss = np.inf
epochs_no_improve = 0

fileout = "models/"+modelfilefront+"avgtime"+str(outputavgtime)+"_allssps_lat"+str(latsel)+"_lon"+str(lonsel)+"_seed"+str(seed)+".pt"

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

valpredclass = np.argmax(valpred,axis=1)
valtrueclass = np.argmax(outputval.numpy(),axis=1)

valacc = np.mean(valpredclass==valtrueclass)
print("best accuracy = "+ str(valacc))
print("on a bg acc of "+ str(valimbalance[0].numpy()) + ":" + str(valimbalance[1].numpy()))

# %%

#plt.scatter(inputvalGMT.numpy().squeeze(),valpred[:,1],marker='.')
#plt.plot(bins[:-1]+0.025,binprobs)

# %%
