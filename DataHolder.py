# DataHolder for grabbing pre-made data

import numpy as np
import glob
import pickle
import time 
import torch
from torch.nn.functional import one_hot

class MPIOutputOnly:
    def __init__(self,params):
        filefront = params["filefront"]
        ssp = params["ssp"]
        outputvar = params["outputvar"]
        self.timerange = params["timerange"]

        outputfile = "data/" + filefront +"summertime_" + ssp + "_" + outputvar + ".pkl"
        outfilecheck = glob.glob(outputfile)

        if len(outfilecheck)==0:
            print('no output file, run DataMaker to make file')
        else:
            with open(outputfile,'rb') as f:
                alloutput = pickle.load(f)
                self.summermean = alloutput[0]
                self.lat = alloutput[1]
                self.lon = alloutput[2]
    


class MPIInputOutput:
    def __init__(self,params):

        # self.inputlength = params["inputlength"]
        # self.outputavgtime = params["outputavgtime"]
        filefront = params["filefront"]
        ssp = params["ssp"]
        inputvar = params["inputvar"]
        outputvar = params["outputvar"]
        self.timerange = params["timerange"]

        inputfile = "data/" + filefront +"annualmean_" + ssp + "_" + inputvar + ".pkl"
        print(inputfile)
        infilecheck = glob.glob(inputfile)

        if len(infilecheck)==0:
            print('no input file, run DataMaker to make file')
        else:
            with open(inputfile,'rb') as f:
                allinput = pickle.load(f)
                self.input_annualmean = allinput[0]
                self.input_lat = allinput[1]
                self.input_lon = allinput[2]

        outputfile = "data/" + filefront +"summertime_" + ssp + "_" + outputvar + ".pkl"
        outfilecheck = glob.glob(outputfile)

        if len(outfilecheck)==0:
            print('no output file, run DataMaker to make file')
        else:
            with open(outputfile,'rb') as f:
                alloutput = pickle.load(f)
                self.output_summermean = alloutput[0]
                self.output_lat = alloutput[1]
                self.output_lon = alloutput[2]

        gmtfile = "data/" + filefront +"annualmeanGMT_" + ssp + "_" + ".pkl"
        gmtfilecheck = glob.glob(gmtfile)

        if len(gmtfilecheck)==0:
            print("no gmt file")
        else:
            with open(gmtfile,'rb') as f:
                self.gmt = pickle.load(f)
    
    def trainvaltest_classifier(self,trainvaltest,historicalera,inputlength,outputlength,tpercentile,latsel,lonsel):

        self.inputlength = inputlength
        self.outputavgtime = outputlength

        if latsel<0:
            self.season = 2
        else:
            self.season = 8

        latindsel = np.argmin(np.abs((self.output_lat-latsel)))
        lonindsel = np.argmin(np.abs((self.output_lon-lonsel)))

        historicalinds = np.asarray(historicalera)-self.timerange[0]

        inputdims = self.input_annualmean.shape[-2:]
        timevec = np.arange(self.timerange[0], self.timerange[1]+1)

        if self.season == 2: 
            timevec = timevec[self.inputlength+1:-1*self.outputavgtime+1]
        
        elif self.season == 8:
            timevec = timevec[self.inputlength:-1*self.outputavgtime+1]

        # first work with gridded input data, 
        # stack it
        stackedinput = stackmatalongdim1(self.input_annualmean,self.inputlength)
        # cut it
        cutinput = stackedinput[:,:-1*(self.outputavgtime)]
        # control for season
        if self.season==2:
            cutinput = cutinput[:,-1]
        # remove mean from sample dimension
        inputmean = np.mean(cutinput,axis=2,keepdims=True)
        anominput = cutinput-inputmean
        # nan out land
        landmask = np.isnan(anominput[0,0,0])
        anominput[:,:,:,landmask] = 0

        # reshape to 4D
        inputtrainfull = np.reshape(anominput[trainvaltest[0]], (len(trainvaltest[0]) * len(timevec), self.inputlength, inputdims[0], inputdims[1]))
        inputvalfull = np.reshape(anominput[trainvaltest[1]], (len(trainvaltest[1]) * len(timevec), self.inputlength, inputdims[0], inputdims[1]))
        inputtestfull = np.reshape(anominput[trainvaltest[2]], (len(trainvaltest[2]) * len(timevec), self.inputlength, inputdims[0], inputdims[1]))

        # now work with GMT data
        # make each sample anomaly from historical era mean
        inputgmtmean = np.mean(self.gmt[:,historicalinds[0]:historicalinds[1]],axis=1,keepdims=True)
        inputgmtanom = self.gmt-inputgmtmean
        # stack it
        stackedgmt = stackmatalongdim1(inputgmtanom,self.inputlength)
        # cut it
        cutgmt = stackedgmt[:,:-1*(self.outputavgtime)]
        # control for season
        if self.season==2:
            cutgmt = cutgmt[:,-1]
        # average over sample dimension but keep that dimension
        avggmt = np.mean(cutgmt,axis=2,keepdims=True)
        # reshape it to 2D
        inputtrainGMTfull = np.reshape(avggmt[trainvaltest[0]], (len(trainvaltest[0]) * len(timevec), 1))
        inputvalGMTfull = np.reshape(avggmt[trainvaltest[1]], (len(trainvaltest[1]) * len(timevec), 1))
        inputtestGMTfull = np.reshape(avggmt[trainvaltest[2]], (len(trainvaltest[2]) * len(timevec), 1))

        # finally, work with output data
        # averaging style of Befort et. al 2025
        summerextreme,one_sigma = onesigma(self.output_summermean[:,:,latindsel,lonindsel], historicalinds, tpercentile)
        # stack it
        stackedoutput = stackmatalongdim1(summerextreme,self.outputavgtime)
        # cut it
        cutoutput = stackedoutput[:,self.inputlength:]
        # control for season
        if self.season==2:
            cutoutput = cutoutput[:,1:]
        # number of extremes in a future period
        nextremes = np.sum(cutoutput,axis=2)

        self.alloutput = nextremes

        # reshape it to 2D
        outputtrainfull = np.reshape(nextremes[trainvaltest[0]], (len(trainvaltest[0]) * len(timevec), 1))
        outputvalfull = np.reshape(nextremes[trainvaltest[1]], (len(trainvaltest[1]) * len(timevec), 1))
        outputtestfull = np.reshape(nextremes[trainvaltest[2]], (len(trainvaltest[2]) * len(timevec), 1))   

        return [inputtrainfull,  inputtrainGMTfull, outputtrainfull,], [inputvalfull, inputvalGMTfull, outputvalfull,], [inputtestfull, inputtestGMTfull,  outputtestfull] 

    def trainvaltest_binaryclassifier(self,trainvaltest,historicalera,inputlength,outputlength,tpercentile,latsel,lonsel):
        
        self.inputlength = inputlength
        self.outputavgtime = outputlength

        if latsel<0:
            self.season = 2
        else:
            self.season = 8

        latindsel = np.argmin(np.abs((self.output_lat-latsel)))
        lonindsel = np.argmin(np.abs((self.output_lon-lonsel)))

        historicalinds = np.asarray(historicalera)-self.timerange[0]

        inputdims = self.input_annualmean.shape[-2:]
        timevec = np.arange(self.timerange[0], self.timerange[1]+1)

        endind = -1*self.outputavgtime+1    
        if endind == 0:
            endind = None

        if self.season == 2: 
            timevec = timevec[self.inputlength+1:endind]
        
        elif self.season == 8:
            timevec = timevec[self.inputlength:endind]

        # first work with gridded input data, 
        # stack it
        stackedinput = stackmatalongdim1(self.input_annualmean,self.inputlength)
        # cut it
        cutinput = stackedinput[:,:-1*(self.outputavgtime)]
        # control for season
        if self.season==2:
            cutinput = cutinput[:,-1]
        # remove mean from sample dimension
        inputmean = np.mean(cutinput,axis=2,keepdims=True)
        anominput = cutinput-inputmean
        # nan out land
        landmask = np.isnan(anominput[0,0,0])
        anominput[:,:,:,landmask] = 0

        # reshape to 4D
        inputtrainfull = np.reshape(anominput[trainvaltest[0]], (len(trainvaltest[0]) * len(timevec), self.inputlength, inputdims[0], inputdims[1]))
        inputvalfull = np.reshape(anominput[trainvaltest[1]], (len(trainvaltest[1]) * len(timevec), self.inputlength, inputdims[0], inputdims[1]))
        inputtestfull = np.reshape(anominput[trainvaltest[2]], (len(trainvaltest[2]) * len(timevec), self.inputlength, inputdims[0], inputdims[1]))

        # now work with GMT data
        # make each sample anomaly from historical era mean
        inputgmtmean = np.mean(self.gmt[:,historicalinds[0]:historicalinds[1]],axis=1,keepdims=True)
        inputgmtanom = self.gmt-inputgmtmean
        # stack it
        stackedgmt = stackmatalongdim1(inputgmtanom,self.inputlength)
        # cut it
        cutgmt = stackedgmt[:,:-1*(self.outputavgtime)]
        # control for season
        if self.season==2:
            cutgmt = cutgmt[:,-1]
        # average over sample dimension but keep that dimension
        avggmt = np.mean(cutgmt,axis=2,keepdims=True)
        # reshape it to 2D
        inputtrainGMTfull = np.reshape(avggmt[trainvaltest[0]], (len(trainvaltest[0]) * len(timevec), 1))
        inputvalGMTfull = np.reshape(avggmt[trainvaltest[1]], (len(trainvaltest[1]) * len(timevec), 1))
        inputtestGMTfull = np.reshape(avggmt[trainvaltest[2]], (len(trainvaltest[2]) * len(timevec), 1))

        # finally, work with output data
        # binary classifier of n year event or not
        # stack it
        stackedoutput = stackmatalongdim1(self.output_summermean[:,:,latindsel,lonindsel],self.outputavgtime)
        # cut it
        cutoutput = stackedoutput[:,self.inputlength:]
        # control for season
        if self.season==2:
            cutoutput = cutoutput[:,1:]
        # number of extremes in a future period
        
        avgsummer = np.mean(cutoutput,axis=2)
        onesigmasummer,onesigmas = onesigma(avgsummer, historicalinds+int(self.outputavgtime/2), tpercentile)
        # nextremes = np.sum(cutoutput,axis=2)

        self.truesummer = avgsummer
        self.alloutput = onesigmasummer
        self.onesigmas = onesigmas

        # reshape it to 2D
        outputtrainfull = np.reshape(onesigmasummer[trainvaltest[0]], (len(trainvaltest[0]) * len(timevec), 1))
        outputvalfull = np.reshape(onesigmasummer[trainvaltest[1]], (len(trainvaltest[1]) * len(timevec), 1))
        outputtestfull = np.reshape(onesigmasummer[trainvaltest[2]], (len(trainvaltest[2]) * len(timevec), 1))   

        return [inputtrainfull,  inputtrainGMTfull, outputtrainfull,], [inputvalfull, inputvalGMTfull, outputvalfull,], [inputtestfull, inputtestGMTfull,  outputtestfull] 

    def trainvaltest_regression(self,trainvaltest,historicalera,inputlength,outputavgtime,latsel,lonsel):

        self.inputlength = inputlength
        self.outputavgtime = outputavgtime

        if latsel<0:
            self.season = 2
        else:
            self.season = 8

        latindsel = np.argmin(np.abs((self.output_lat-latsel)))
        lonindsel = np.argmin(np.abs((self.output_lon-lonsel)))

        historicalinds = np.asarray(historicalera)-self.timerange[0]

        inputdims = self.input_annualmean.shape[-2:]
        timevec = np.arange(self.timerange[0], self.timerange[1]+1)

        if self.season == 2: 
            timevec = timevec[self.inputlength+1:-1*self.outputavgtime+1]
        
        elif self.season == 8:
            timevec = timevec[self.inputlength:-1*self.outputavgtime+1]

        # first work with gridded input data, 
        # stack it
        stackedinput = stackmatalongdim1(self.input_annualmean,self.inputlength)
        # cut it
        cutinput = stackedinput[:,:-1*(self.outputavgtime)]
        # control for season
        if self.season==2:
            cutinput = cutinput[:,-1]
        # remove mean from sample dimension
        inputmean = np.mean(cutinput,axis=2,keepdims=True)
        anominput = cutinput-inputmean
        # nan out land
        landmask = np.isnan(anominput[0,0,0])
        anominput[:,:,:,landmask] = 0

        # reshape to 4D
        inputtrainfull = np.reshape(anominput[trainvaltest[0]], (len(trainvaltest[0]) * len(timevec), self.inputlength, inputdims[0], inputdims[1]))
        inputvalfull = np.reshape(anominput[trainvaltest[1]], (len(trainvaltest[1]) * len(timevec), self.inputlength, inputdims[0], inputdims[1]))
        inputtestfull = np.reshape(anominput[trainvaltest[2]], (len(trainvaltest[2]) * len(timevec), self.inputlength, inputdims[0], inputdims[1]))

        # now work with GMT data
        # make each sample anomaly from historical era mean
        inputgmtmean = np.mean(self.gmt[:,historicalinds[0]:historicalinds[1]],axis=1,keepdims=True)
        inputgmtanom = self.gmt-inputgmtmean
        # stack it
        stackedgmt = stackmatalongdim1(inputgmtanom,self.inputlength)
        # cut it
        cutgmt = stackedgmt[:,:-1*(self.outputavgtime)]
        # control for season
        if self.season==2:
            cutgmt = cutgmt[:,-1]
        # average over sample dimension but keep that dimension
        avggmt = np.mean(cutgmt,axis=2,keepdims=True)
        # reshape it to 2D
        inputtrainGMTfull = np.reshape(avggmt[trainvaltest[0]], (len(trainvaltest[0]) * len(timevec), 1))
        inputvalGMTfull = np.reshape(avggmt[trainvaltest[1]], (len(trainvaltest[1]) * len(timevec), 1))
        inputtestGMTfull = np.reshape(avggmt[trainvaltest[2]], (len(trainvaltest[2]) * len(timevec), 1))

        # finally, work with output data
        # subtract summer histroical era temperature to make anomaly timeseries
        outputmean = np.mean(self.output_summermean[:,:,latindsel,lonindsel],axis=1,keepdims=True)
        outputanom = self.output_summermean[:,:,latindsel,lonindsel]-outputmean
        # stack it
        stackedoutput = stackmatalongdim1(outputanom,self.outputavgtime)
        # cut it
        cutoutput = stackedoutput[:,self.inputlength:]
        # control for season
        if self.season==2:
            cutoutput = cutoutput[:,1:]

        # average to make future means
        avgoutput = np.mean(cutoutput,axis=2)
        self.alloutput = avgoutput

        # reshape it to 2D
        outputtrainfull = np.reshape(avgoutput[trainvaltest[0]], (len(trainvaltest[0]) * len(timevec), 1))
        outputvalfull = np.reshape(avgoutput[trainvaltest[1]], (len(trainvaltest[1]) * len(timevec), 1))
        outputtestfull = np.reshape(avgoutput[trainvaltest[2]], (len(trainvaltest[2]) * len(timevec), 1))

        return [inputtrainfull,  inputtrainGMTfull, outputtrainfull,], [inputvalfull, inputvalGMTfull, outputvalfull,], [inputtestfull, inputtestGMTfull,  outputtestfull] 

def tensortime_multihot(listofmatrices):

    inputdata = listofmatrices[0]
    inputgmt = listofmatrices[1]
    outputdata = listofmatrices[2]

    inputdata_t = torch.tensor(inputdata,dtype=torch.float32)
    inputgmt_t = torch.tensor(inputgmt,dtype=torch.float32)

    outputmultihotencoded = multihotencode(outputdata)
    outputdata_t = torch.tensor(outputmultihotencoded,dtype=torch.float32)

    return inputdata_t, inputgmt_t, outputdata_t

def tensortime_onehot(listofmatrices,nclasses=6):

    inputdata = listofmatrices[0]
    inputgmt = listofmatrices[1]
    outputdata = listofmatrices[2]

    inputdata_t = torch.tensor(inputdata,dtype=torch.float32)
    inputgmt_t = torch.tensor(inputgmt,dtype=torch.float32)

    outputdata = torch.tensor(outputdata.squeeze(),dtype=torch.int64)
    outputonehotencoded = one_hot(outputdata, num_classes=nclasses) 
    outputdata_t = torch.tensor(outputonehotencoded,dtype=torch.float32)

    return inputdata_t, inputgmt_t, outputdata_t

def tensortime_regression(listofmatrices):

    inputdata = listofmatrices[0]
    inputgmt = listofmatrices[1]
    outputdata = listofmatrices[2]

    inputdata_t = torch.tensor(inputdata,dtype=torch.float32)
    inputgmt_t = torch.tensor(inputgmt,dtype=torch.float32)
    outputdata_t = torch.tensor(outputdata,dtype=torch.float32)

    return inputdata_t, inputgmt_t, outputdata_t

def stackmatalongdim1(mat,stacklength):

    matshape = mat.shape
    if len(matshape) == 4:
        newshape = (matshape[0],matshape[1]-stacklength+1,stacklength,matshape[2],matshape[3])
    elif len(matshape) == 2:
        newshape = (matshape[0],matshape[1]-stacklength+1,stacklength)
    else:
        raise ValueError("Input matrix must be 2D or 4D")

    newmat = np.empty(newshape)

    for istack in range(newmat.shape[1]):

        newmat[:,istack] = mat[:,istack:istack+stacklength]

    return newmat        

def onesigma(mat,historicalinds,tpercentile):

    one_sigma = np.percentile(mat[:,historicalinds[0]:historicalinds[1]],tpercentile,axis=1) # one sigma threshold in each member
    # one_sigma = np.max(mat[:,historicalinds[0]:historicalinds[1]],axis=1)
    summer_extreme = 1.*(mat>one_sigma[:,np.newaxis]) # one zeros for if event happens or not

    return summer_extreme,one_sigma

def multihotencode(mat):

    mat = np.squeeze(mat)
    outdims = (mat.shape[0], int(np.max(mat)))
    multihotencoded = np.empty(outdims)

    for i in range(outdims[0]):
        encoding = np.zeros(outdims[1])
        encoding[:int(mat[i])] = 1
        multihotencoded[i] = encoding
    
    return multihotencoded