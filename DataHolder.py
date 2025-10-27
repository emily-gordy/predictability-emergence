# DataHolder for grabbing pre-made data

import numpy as np
import glob
import pickle
import time 
import torch
from torch.nn.functional import one_hot


endhist = 2014

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


class MPIInputOutput_SSPlist:
    def __init__(self,params,ssplist):

        filefront = params["filefront"]
        inputvar = params["inputvar"]
        outputvar = params["outputvar"]
        self.timerange = params["timerange"]
        self.ssplist = ssplist

        allinput = []
        alloutput = []
        allGMT = []

        for issp, ssp in enumerate(ssplist):

            inputfile = "data/" + filefront +"annualmean_" + ssp + "_" + inputvar + "_"+str(self.timerange[0])+"-"+str(self.timerange[1])+".pkl"
            print(inputfile)
            infilecheck = glob.glob(inputfile)

            if len(infilecheck)==0:
                print('no input file, run DataMaker to make file')
            else:
                with open(inputfile,'rb') as f:
                    inputs = pickle.load(f)
                    input_annualmean = inputs[0]
                    self.input_lat = inputs[1]
                    self.input_lon = inputs[2]
            allinput.append(input_annualmean)

            outputfile = "data/" + filefront +"summertime_" + ssp + "_" + outputvar + "_"+str(self.timerange[0])+"-"+str(self.timerange[1])+".pkl"
            outfilecheck = glob.glob(outputfile)

            if len(outfilecheck)==0:
                print('no output file, run DataMaker to make file')
            else:
                with open(outputfile,'rb') as f:
                    output = pickle.load(f)
                    output_summermean = output[0]
                    self.output_lat = output[1]
                    self.output_lon = output[2]

            alloutput.append(output_summermean)

            gmtfile = "data/" + filefront +"annualmeanGMT_" + ssp + "_"+str(self.timerange[0])+"-"+str(self.timerange[1])+".pkl"
            gmtfilecheck = glob.glob(gmtfile)

            if len(gmtfilecheck)==0:
                print("no gmt file")
            else:
                with open(gmtfile,'rb') as f:
                    gmt = pickle.load(f)

            allGMT.append(gmt)

        self.allinput = allinput
        self.alloutput = alloutput
        self.allGMT = allGMT

    def trainvaltest_binaryclassifier(self,trainvaltest,historicalera,inputlength,outputlength,tpercentile,latsel,lonsel):
        
        self.inputlength = inputlength
        self.outputavgtime = outputlength

        allsummer = []

        if latsel<0:
            self.season = 2
        else:
            self.season = 8

        latindsel = np.argmin(np.abs((self.output_lat-latsel)))
        lonindsel = np.argmin(np.abs((self.output_lon-lonsel)))

        historicalinds = np.asarray(historicalera)-self.timerange[0]

        timevecfull = np.arange(self.timerange[0], self.timerange[1]+1)

        inputdims = self.allinput[0].shape[-2:]

        MPIhistorical = [0,endhist-timevecfull[0]+self.outputavgtime]

        input_annualmean = self.allinput[0] # grab the first ssp, doesn't really matter which
        input_hist = input_annualmean[:,MPIhistorical[0]:MPIhistorical[1]]
        # first work with gridded input data, 
        # stack it
        stackedinput = stackmatalongdim1(input_hist,self.inputlength)
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

        inputtrainfull = np.reshape(anominput[trainvaltest[0]], (len(trainvaltest[0]) * anominput.shape[1], self.inputlength, inputdims[0], inputdims[1]))
        inputvalfull = np.reshape(anominput[trainvaltest[1]], (len(trainvaltest[1]) * anominput.shape[1], self.inputlength, inputdims[0], inputdims[1]))
        inputtestfull = np.reshape(anominput[trainvaltest[2]], (len(trainvaltest[2]) * anominput.shape[1], self.inputlength, inputdims[0], inputdims[1]))

        gmt = self.allGMT[0]
        gmt_hist = gmt[:,MPIhistorical[0]:MPIhistorical[1]]
        # now work with GMT data
        # make each sample anomaly from historical era mean
        inputgmtmean = np.mean(gmt_hist[:,historicalinds[0]:historicalinds[1]],axis=1,keepdims=True)
        
        # print('checking size of inputgmtmean')
        # print(str(inputgmtmean.shape))

        inputgmtanom = gmt_hist-inputgmtmean
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
        inputtrainGMTfull = np.reshape(avggmt[trainvaltest[0]], (len(trainvaltest[0]) * avggmt.shape[1], 1))
        inputvalGMTfull = np.reshape(avggmt[trainvaltest[1]], (len(trainvaltest[1]) * avggmt.shape[1], 1))
        inputtestGMTfull = np.reshape(avggmt[trainvaltest[2]], (len(trainvaltest[2]) * avggmt.shape[1], 1))

        output_summermean = self.alloutput[0]
        output_summermean_hist = output_summermean[:,MPIhistorical[0]:MPIhistorical[1]]
        # finally, work with output data
        # binary classifier of n year event or not
        # stack it
        stackedoutput = stackmatalongdim1(output_summermean_hist[:,:,latindsel,lonindsel],self.outputavgtime)
        # cut it
        cutoutput = stackedoutput[:,self.inputlength:]
        # control for season
        if self.season==2:
            cutoutput = cutoutput[:,1:]
        # number of extremes in a future period
        
        avgsummer = np.mean(cutoutput,axis=2)
        onesigmasummer,onesigmas = onesigma(avgsummer, historicalinds+int(self.outputavgtime/2), tpercentile)

        allsummer.append(avgsummer)

        # print('checking size of onesigmas')
        # print(str(onesigmas.shape))

        # reshape it to 2D
        outputtrainfull = np.reshape(onesigmasummer[trainvaltest[0]], (len(trainvaltest[0]) * onesigmasummer.shape[1], 1))
        outputvalfull = np.reshape(onesigmasummer[trainvaltest[1]], (len(trainvaltest[1]) * onesigmasummer.shape[1], 1))
        outputtestfull = np.reshape(onesigmasummer[trainvaltest[2]], (len(trainvaltest[2]) * onesigmasummer.shape[1], 1))   

        endind = -1*self.outputavgtime+1    
        if endind == 0:
            endind = None

        for issp, ssp in enumerate(self.ssplist):

            # print('working on ' + ssp)
            
            MPIfuture = [endhist-timevecfull[0]-self.inputlength,endind]

            input_annualmean = self.allinput[issp]
            input_future = input_annualmean[:,MPIfuture[0]:MPIfuture[1]]
            # first work with gridded input data, 
            # stack it
            stackedinput = stackmatalongdim1(input_future,self.inputlength)
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
            anomreshape_train = np.reshape(anominput[trainvaltest[0]], (len(trainvaltest[0]) * anominput.shape[1], self.inputlength, inputdims[0], inputdims[1]))
            anomreshape_val = np.reshape(anominput[trainvaltest[1]], (len(trainvaltest[1]) * anominput.shape[1], self.inputlength, inputdims[0], inputdims[1]))
            anomreshape_test = np.reshape(anominput[trainvaltest[2]], (len(trainvaltest[2]) * anominput.shape[1], self.inputlength, inputdims[0], inputdims[1]))

            inputtrainfull = np.append(inputtrainfull,anomreshape_train,axis=0)
            inputvalfull = np.append(inputvalfull,anomreshape_val,axis=0)
            inputtestfull = np.append(inputtestfull,anomreshape_test,axis=0)

            gmt = self.allGMT[issp]
            gmt_future = gmt[:,MPIfuture[0]:MPIfuture[1]]
            # now work with GMT data

            inputgmtanom = gmt_future-inputgmtmean #use gmt mean from historical (already calculated)
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
            avggmtreshape_train = np.reshape(avggmt[trainvaltest[0]], (len(trainvaltest[0]) * avggmt.shape[1], 1))
            avggmtreshape_val = np.reshape(avggmt[trainvaltest[1]], (len(trainvaltest[1]) * avggmt.shape[1], 1))
            avggmtreshape_test = np.reshape(avggmt[trainvaltest[2]], (len(trainvaltest[2]) * avggmt.shape[1], 1))

            inputtrainGMTfull = np.append(inputtrainGMTfull,avggmtreshape_train,axis=0)
            inputvalGMTfull = np.append(inputvalGMTfull,avggmtreshape_val,axis=0)
            inputtestGMTfull = np.append(inputtestGMTfull,avggmtreshape_test,axis=0)

            output_summermean = self.alloutput[issp]
            output_summer_hist = output_summermean[:,MPIfuture[0]:MPIfuture[1]]
            # finally, work with output data
            # binary classifier of n year event or not
            # stack it
            stackedoutput = stackmatalongdim1(output_summer_hist[:,:,latindsel,lonindsel],self.outputavgtime)
            # cut it
            cutoutput = stackedoutput[:,self.inputlength:]
            # control for season
            if self.season==2:
                cutoutput = cutoutput[:,1:]
            # number of extremes in a future period
            
            avgsummer = np.mean(cutoutput,axis=2)
            # onesigmasummer,onesigmas = onesigma(avgsummer, historicalinds+int(self.outputavgtime/2), tpercentile)
            # nextremes = np.sum(cutoutput,axis=2)
            onesigmasummer = 1.*(avgsummer>onesigmas[:,np.newaxis])

            allsummer.append(avgsummer)

            # reshape it to 2D

            summerreshape_train = np.reshape(onesigmasummer[trainvaltest[0]], (len(trainvaltest[0]) * onesigmasummer.shape[1], 1))
            summerreshape_val = np.reshape(onesigmasummer[trainvaltest[1]], (len(trainvaltest[1]) * onesigmasummer.shape[1], 1))
            summerreshape_test = np.reshape(onesigmasummer[trainvaltest[2]], (len(trainvaltest[2]) * onesigmasummer.shape[1], 1))

            outputtrainfull = np.append(outputtrainfull,summerreshape_train,axis=0)
            outputvalfull = np.append(outputvalfull,summerreshape_val,axis=0)
            outputtestfull = np.append(outputtestfull,summerreshape_test,axis=0)

        self.truesummer = allsummer
        self.onesigmas = onesigmas

        return [inputtrainfull,  inputtrainGMTfull, outputtrainfull,], [inputvalfull, inputvalGMTfull, outputvalfull,], [inputtestfull, inputtestGMTfull,  outputtestfull] 


    def calculate_nulls(self,trainvaltest):
        
        onesigmas = self.onesigmas

        summerloop = self.truesummer[0]
        summerboo = 1*(summerloop>onesigmas[:,np.newaxis])

        meanpred = np.mean(summerboo[trainvaltest[0]],axis=0)
        gmtpred = np.round(meanpred)

        nullval = np.tile(gmtpred,(len(trainvaltest[1])))
        nulltest = np.tile(gmtpred,(len(trainvaltest[2])))

        for issp in range(len(self.ssplist)):

            summerloop = self.truesummer[issp+1]
            summerboo = 1*(summerloop>onesigmas[:,np.newaxis])

            meanpred = np.mean(summerboo[trainvaltest[0]],axis=0)
            gmtpred = np.round(meanpred)

            gmtpredval = np.tile(gmtpred,(len(trainvaltest[1])))
            gmtpredtest = np.tile(gmtpred,(len(trainvaltest[2])))

            nullval = np.append(nullval,gmtpredval,axis=0)
            nulltest = np.append(nulltest,gmtpredtest,axis=0)
        
        return nullval,nulltest

    def trainvaltest_recordmax(self,trainvaltest,experimentera,baselineera,inputlength,outputlength,latsel,lonsel):

        self.inputlength = inputlength
        self.outputavgtime = outputlength

        allsummer = []

        if latsel<0:
            self.season = 2
        else:
            self.season = 8

        latindsel = np.argmin(np.abs((self.output_lat-latsel)))
        lonindsel = np.argmin(np.abs((self.output_lon-lonsel)))

        gmthistorical = [1960,1990]
        gmthistoricalinds = [gmthistorical[0]-self.timerange[0],gmthistorical[1]-self.timerange[0]]

        print('baseline for gmt is '+ str(gmthistoricalinds[0]) + ' ' + str(gmthistoricalinds[1]))

        # historicalinds = np.asarray(historicalera)-self.timerange[0]

        # timevecfull = np.arange(self.timerange[0], self.timerange[1]+1)

        inputdims = self.allinput[0].shape[-2:]

        historicalinds = [experimentera[0]-self.timerange[0],endhist-self.timerange[0]+self.outputavgtime]
        print('experiment era in the historical is '+ str(historicalinds[0]) + ' ' + str(historicalinds[1]))

        input_annualmean = self.allinput[0] # grab the first ssp, doesn't really matter which
        input_hist = input_annualmean[:,historicalinds[0]:historicalinds[1]]
        # first work with gridded input data, 
        # stack it
        stackedinput = stackmatalongdim1(input_hist,self.inputlength)
        # cut it
        cutinput = stackedinput[:,:-1*(self.outputavgtime)]
        # control for season
        if self.season==2:
            cutinput = cutinput[:,:-1]
        # remove mean from sample dimension
        inputmean = np.mean(cutinput,axis=2,keepdims=True)
        anominput = cutinput-inputmean
        # nan out land
        landmask = np.isnan(anominput[0,0,0])
        anominput[:,:,:,landmask] = 0

        inputtrainfull = np.reshape(anominput[trainvaltest[0]], (len(trainvaltest[0]) * anominput.shape[1], self.inputlength, inputdims[0], inputdims[1]))
        inputvalfull = np.reshape(anominput[trainvaltest[1]], (len(trainvaltest[1]) * anominput.shape[1], self.inputlength, inputdims[0], inputdims[1]))
        inputtestfull = np.reshape(anominput[trainvaltest[2]], (len(trainvaltest[2]) * anominput.shape[1], self.inputlength, inputdims[0], inputdims[1]))

        gmt = self.allGMT[0]
        gmt_hist = gmt[:,historicalinds[0]:historicalinds[1]]
        # now work with GMT data
        # make each sample anomaly from historical era mean
        inputgmtmean = np.mean(gmt[:,gmthistoricalinds[0]:gmthistoricalinds[1]],axis=1,keepdims=True)
        
        # print('checking size of inputgmtmean')
        # print(str(inputgmtmean.shape))

        inputgmtanom = gmt_hist-inputgmtmean
        # stack it
        stackedgmt = stackmatalongdim1(inputgmtanom,self.inputlength)
        # cut it
        cutgmt = stackedgmt[:,:-1*(self.outputavgtime)]
        # control for season
        if self.season==2:
            cutgmt = cutgmt[:,:-1]
        # average over sample dimension but keep that dimension
        avggmt = np.mean(cutgmt,axis=2,keepdims=True)
        # reshape it to 2D
        inputtrainGMTfull = np.reshape(avggmt[trainvaltest[0]], (len(trainvaltest[0]) * avggmt.shape[1], 1))
        inputvalGMTfull = np.reshape(avggmt[trainvaltest[1]], (len(trainvaltest[1]) * avggmt.shape[1], 1))
        inputtestGMTfull = np.reshape(avggmt[trainvaltest[2]], (len(trainvaltest[2]) * avggmt.shape[1], 1))

        # output_summermean = self.alloutput[0]
        # calculate record summers

        # exprange = [experiment_era[0]-timerange[0],experiment_era[1]-timerange[0]]
        baselineindices = [baselineera[0]-self.timerange[0],baselineera[1]-self.timerange[0]]
        print('baseline for temp records is '+ str(baselineindices[0]) + ' ' + str(baselineindices[1]))

        recordtemps = np.zeros((len(self.ssplist),self.alloutput[0].shape[0],self.alloutput[0].shape[1]))

        for issp,ssp in enumerate(self.ssplist):

            histsel = self.alloutput[issp]
            histsel = histsel[:,:,latindsel,lonindsel]

            baseline = np.max(histsel[:,baselineindices[0]:baselineindices[1]],axis=1)
            
            for iens in range(histsel.shape[0]):

                recordtemps[issp,iens,:] = find_records(histsel[iens,:],baseline[iens])

        self.recordtemps = recordtemps

        # finally, work with output data
        # binary classifier of record summer occurs or not

        output_recordtemps_hist = recordtemps[0,:,historicalinds[0]:historicalinds[1]]
        # stack it
        stackedoutput = stackmatalongdim1(output_recordtemps_hist,self.outputavgtime)

         # cut it
        cutoutput = stackedoutput[:,self.inputlength:]
        # control for season
        if self.season==2:
            cutoutput = cutoutput[:,1:]
        # number of extremes in a future period
        
        nevents = np.sum(cutoutput,axis=2)
        outputrecordsummer = 1*(nevents>0) # any number >0 is a yes
        allsummer.append(nevents)

        # reshape it to 2D
        outputtrainfull = np.reshape(outputrecordsummer[trainvaltest[0]], (len(trainvaltest[0]) * outputrecordsummer.shape[1], 1))
        outputvalfull = np.reshape(outputrecordsummer[trainvaltest[1]], (len(trainvaltest[1]) * outputrecordsummer.shape[1], 1))
        outputtestfull = np.reshape(outputrecordsummer[trainvaltest[2]], (len(trainvaltest[2]) * outputrecordsummer.shape[1], 1))   

        endind = -1*self.outputavgtime+1    
        if endind == 0:
            endind = None

        futureinds = [endhist-self.timerange[0]-self.inputlength,endind]
        print('indices of future period are ' + str(futureinds[0]) + ' ' + str(futureinds[1]))

        for issp, ssp in enumerate(self.ssplist):

            print('working on ' + ssp)

            input_annualmean = self.allinput[issp]
            input_future = input_annualmean[:,futureinds[0]:futureinds[1]]
            # first work with gridded input data, 
            # stack it
            stackedinput = stackmatalongdim1(input_future,self.inputlength)
            # cut it
            cutinput = stackedinput[:,:-1*(self.outputavgtime)]
            # control for season
            if self.season==2:
                cutinput = cutinput[:,:-1]
            # remove mean from sample dimension
            inputmean = np.mean(cutinput,axis=2,keepdims=True)
            anominput = cutinput-inputmean
            # nan out land
            landmask = np.isnan(anominput[0,0,0])
            anominput[:,:,:,landmask] = 0

            # reshape to 4D
            anomreshape_train = np.reshape(anominput[trainvaltest[0]], (len(trainvaltest[0]) * anominput.shape[1], self.inputlength, inputdims[0], inputdims[1]))
            anomreshape_val = np.reshape(anominput[trainvaltest[1]], (len(trainvaltest[1]) * anominput.shape[1], self.inputlength, inputdims[0], inputdims[1]))
            anomreshape_test = np.reshape(anominput[trainvaltest[2]], (len(trainvaltest[2]) * anominput.shape[1], self.inputlength, inputdims[0], inputdims[1]))

            inputtrainfull = np.append(inputtrainfull,anomreshape_train,axis=0)
            inputvalfull = np.append(inputvalfull,anomreshape_val,axis=0)
            inputtestfull = np.append(inputtestfull,anomreshape_test,axis=0)

            gmt = self.allGMT[issp]
            gmt_future = gmt[:,futureinds[0]:futureinds[1]]
            # now work with GMT data

            inputgmtanom = gmt_future-inputgmtmean #use gmt mean from historical (already calculated)
            # stack it
            stackedgmt = stackmatalongdim1(inputgmtanom,self.inputlength)
            # cut it
            cutgmt = stackedgmt[:,:-1*(self.outputavgtime)]
            # control for season
            if self.season==2:
                cutgmt = cutgmt[:,:-1]
            # average over sample dimension but keep that dimension
            avggmt = np.mean(cutgmt,axis=2,keepdims=True)

            # reshape it to 2D
            avggmtreshape_train = np.reshape(avggmt[trainvaltest[0]], (len(trainvaltest[0]) * avggmt.shape[1], 1))
            avggmtreshape_val = np.reshape(avggmt[trainvaltest[1]], (len(trainvaltest[1]) * avggmt.shape[1], 1))
            avggmtreshape_test = np.reshape(avggmt[trainvaltest[2]], (len(trainvaltest[2]) * avggmt.shape[1], 1))

            inputtrainGMTfull = np.append(inputtrainGMTfull,avggmtreshape_train,axis=0)
            inputvalGMTfull = np.append(inputvalGMTfull,avggmtreshape_val,axis=0)
            inputtestGMTfull = np.append(inputtestGMTfull,avggmtreshape_test,axis=0)

            # finally, work with output data
            # binary classifier of n year event or not
            # stack it

            output_recordtemps_future = recordtemps[issp,:,futureinds[0]:futureinds[1]]
            # stack it
            stackedoutput = stackmatalongdim1(output_recordtemps_future,self.outputavgtime)
            # cut it
            cutoutput = stackedoutput[:,self.inputlength:]
            # control for season
            if self.season==2:
                cutoutput = cutoutput[:,1:]
            # number of extremes in a future period

            nevents = np.sum(cutoutput,axis=2)
            outputrecordsummer = 1*(nevents>0) # any number >0 is a yes

            allsummer.append(nevents)

            # reshape it to 2D

            summerreshape_train = np.reshape(outputrecordsummer[trainvaltest[0]], (len(trainvaltest[0]) * outputrecordsummer.shape[1], 1))
            summerreshape_val = np.reshape(outputrecordsummer[trainvaltest[1]], (len(trainvaltest[1]) * outputrecordsummer.shape[1], 1))
            summerreshape_test = np.reshape(outputrecordsummer[trainvaltest[2]], (len(trainvaltest[2]) * outputrecordsummer.shape[1], 1))

            outputtrainfull = np.append(outputtrainfull,summerreshape_train,axis=0)
            outputvalfull = np.append(outputvalfull,summerreshape_val,axis=0)
            outputtestfull = np.append(outputtestfull,summerreshape_test,axis=0)

        self.truesummer = allsummer

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

def find_records(temps,baseline):
    # Compare each value to the maximum of all previous values
    records = np.zeros(len(temps), dtype=bool)
    # records[0] = True  # First value is always a "record"
    current_max = np.copy(baseline)
    
    for i, temp in enumerate(temps):
        if temp > current_max:
            records[i] = 1
            current_max = temp

    return records