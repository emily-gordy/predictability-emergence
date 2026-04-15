"""Some helper functions for plotting and baseline models"""

import sys
import os

sys.path.append(os.path.abspath('./predictability_emergence')) 

import DataMakar
import DataHolder
import buildmodel
import helpers

import importlib as imp
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

import torch.optim.lr_scheduler as lr_scheduler
import time

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

import glob
import json
import pickle

import matplotlib as mpl
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available() & torch.backends.mps.is_built():
    device = 'mps'
else:
    device='cpu'

class RecordMax:
    def __init__(self,analysisparams):
        
        self.modelfilefront = "MPI_recordtemp_"

        self.trainvaltest = analysisparams["trainvaltest"]
        self.experiment_era = analysisparams["experiment_era"]
        self.baselineera = analysisparams["baselineera"]
        self.inputlength = analysisparams["inputlength"]
        self.outputavgtime = analysisparams["outputavgtime"]
        self.ssplist = analysisparams["ssps"]
        self.ssplistplot = ["hist"]+ self.ssplist 
        self.experiment_era_obs = analysisparams["experiment_era_obs"]
        self.baselineera_obs = analysisparams["baselineera_obs"]
        self.obstimerange = analysisparams["obstimerange"]

    def logistic_regression(self,AllData,sspkeep):
        
        alltrain, _, alltest = AllData.trainvaltest_recordmax_withrecordmax(self.trainvaltest,self.experiment_era,self.baselineera,self.inputlength,self.outputavgtime,-20,0)

        # _, inputtrainGMT, _  = DataHolder.tensortime_onehot_withrecordmax(alltrain,nclasses=2)
        inputtrainGMT = alltrain[1]
        inputtestGMT = alltest[1]
        # _, inputtestGMT, _ = DataHolder.tensortime_onehot_withrecordmax(alltest,nclasses=2)

        inputssptrain,_,inputssptest = AllData.trainvaltest_recordmax_ssps(self.trainvaltest,self.experiment_era,self.inputlength,self.outputavgtime,-20)

        inputssptrain = np.round((inputssptrain.squeeze()+0.2)*10)
        # inputsspval = np.round((inputsspval.squeeze()+0.2)*10)
        inputssptest = np.round((inputssptest.squeeze()+0.2)*10) # EMILY STOP COMMENTING THIS OUT IT IS NECESSARY YOU IDIOT

        regenerate = False

        lr_first_positive = np.empty((5,len(AllData.output_lat),len(AllData.output_lon)))+np.nan
        lr_first_correct = np.empty((5,len(AllData.output_lat),len(AllData.output_lon)))+np.nan
        lr_first_confident = np.empty((5,len(AllData.output_lat),len(AllData.output_lon)))+np.nan

        lr_accuracy_ssp = np.empty((5,len(AllData.output_lat),len(AllData.output_lon),20))+np.nan
        bs_lr_ssp = np.empty((5,len(AllData.output_lat),len(AllData.output_lon)))+np.nan

        lr_accuracy_all = np.empty((len(AllData.output_lat),len(AllData.output_lon),20))+np.nan
        bs_lr_all = np.empty((len(AllData.output_lat),len(AllData.output_lon)))+np.nan

        lr_coeffs = np.empty((len(AllData.output_lat),len(AllData.output_lon)))+np.nan

        for ilatsel,latsel in enumerate(AllData.output_lat):
            print(latsel)

            if (latsel>=0) & (regenerate==False):

                alltrain, _, alltest = AllData.trainvaltest_recordmax_withrecordmax(self.trainvaltest,self.experiment_era,self.baselineera,self.inputlength,self.outputavgtime,latsel,0)
                inputssptrain,_,inputssptest = AllData.trainvaltest_recordmax_ssps(self.trainvaltest,self.experiment_era,self.inputlength,self.outputavgtime,latsel)

                inputtrainGMT = alltrain[1]
                inputtestGMT = alltest[1]   

                inputssptrain,_,inputssptest = AllData.trainvaltest_recordmax_ssps(self.trainvaltest,self.experiment_era,self.inputlength,self.outputavgtime,0)

                inputssptrain = np.round((inputssptrain.squeeze()+0.2)*10)
                inputssptest = np.round((inputssptest.squeeze()+0.2)*10)

                regenerate = True

            lr_pred_file = "predictions/"+self.modelfilefront+"avgtime_"+str(self.outputavgtime)+"_allssps_lat_"+str(latsel)+"_logisticregression.pkl"
            if len(glob.glob(lr_pred_file))==0:
                lr_preds = np.empty((2,36,len(inputtestGMT.squeeze()))) + np.nan

            for ilonsel,lonsel in enumerate(AllData.output_lon):

                outputtrainall, _, outputtestall = AllData.trainvaltest_recordmax_outputonly(self.trainvaltest,self.experiment_era,self.baselineera,self.inputlength,self.outputavgtime,latsel,lonsel)

                if (np.mean(outputtrainall)!=0) & (np.mean(outputtrainall)!=1):

                    #grab hist only
                    inputtrainGMT = inputtrainGMT[:,[0]]
                    inputtestGMT = inputtestGMT[:,[0]]

                    histtraingmt = inputtrainGMT[inputssptrain==0]
                    histtestgmt = inputtestGMT[inputssptest==0]

                    outputtrainhist = outputtrainall[inputssptrain==0]
                    outputtesthist = outputtestall[inputssptest==0]

                    ssplabelhist = inputssptest[inputssptest==0]

                    all_lr_pred = np.empty(1)
                    all_lr_conf = np.empty(1)
                    all_lr_true = np.empty(1)

                    for issp in range(1,5):

                        inputtraingmtssp = inputtrainGMT[inputssptrain==issp]
                        allinputtraingmt = np.concatenate((histtraingmt,inputtraingmtssp),axis=0)

                        outputssptrain = outputtrainall[inputssptrain==issp]
                        alloutputtrain = np.concatenate((outputtrainhist,outputssptrain),axis=0)

                        inputtestgmtssp = inputtestGMT[inputssptest==issp]
                        allinputtestgmt = np.concatenate((histtestgmt,inputtestgmtssp),axis=0)

                        outputssptest = outputtestall[inputssptest==issp]
                        alloutputtest = np.concatenate((outputtesthist,outputssptest),axis=0)

                        ssplabelssp = inputssptest[inputssptest==issp]
                        ssplabelsall = np.concatenate((ssplabelhist,ssplabelssp),axis=0)

                        # get the CNN predictions

                        model = LogisticRegression(random_state=42)

                        # Fit the model
                        model.fit(allinputtraingmt, alloutputtrain.squeeze())

                        y_pred_all = model.predict(allinputtestgmt)
                        y_pred_all_prob = model.predict_proba(allinputtestgmt)
                        y_conf_all = np.max(y_pred_all_prob,axis=1)

                        lr_accuracy_ssp[issp,ilatsel,ilonsel] = helpers.confacc(y_pred_all.squeeze(),alloutputtest.squeeze(),y_conf_all)

                        lr_first_positive[issp,ilatsel,ilonsel] = helpers.firstpositive(y_pred_all_prob[:,1],allinputtestgmt.squeeze())
                        lr_first_correct[issp,ilatsel,ilonsel] = helpers.firstcorrect(y_pred_all_prob[:,1],alloutputtest.squeeze(),allinputtestgmt.squeeze())
                        lr_first_confident[issp,ilatsel,ilonsel] = helpers.firstconfident(y_pred_all_prob[:,1],allinputtestgmt.squeeze())

                        # save predictions of all ssps and only hist for selected ssp

                        all_lr_pred = np.append(all_lr_pred,y_pred_all_prob[ssplabelsall==issp,1])
                        all_lr_conf = np.append(all_lr_conf,np.max(y_pred_all_prob[ssplabelsall==issp,:],axis=1))
                        all_lr_true = np.append(all_lr_true,outputssptest)

                        if issp == sspkeep:

                            lrmodelsavefile = "lr_models/"+ self.modelfilefront + "LogisticRegression_avgtime_"+str(self.outputavgtime)+"_lat_"+str(latsel)+"_lon_"+str(lonsel)+".joblib"
                            joblib.dump(model, lrmodelsavefile)

                            all_lr_pred = np.append(all_lr_pred,y_pred_all_prob[ssplabelsall==0,1])
                            all_lr_conf = np.append(all_lr_conf,np.max(y_pred_all_prob[ssplabelsall==0,:],axis=1))
                            all_lr_true = np.append(all_lr_true,outputtesthist)

                            y_pred_hist_prob = model.predict_proba(histtestgmt)
                            lr_first_correct[0,ilatsel,ilonsel] = helpers.firstcorrect(y_pred_hist_prob[:,1],outputtesthist.squeeze(),histtestgmt.squeeze())

                            lr_coeffs[ilatsel,ilonsel] = model.coef_[0]

                    if len(glob.glob(lr_pred_file))==0:
                        lr_preds[0,ilonsel] = all_lr_pred.squeeze()[1:]
                        lr_preds[1,ilonsel] = all_lr_true.squeeze()[1:]

                    lr_accuracy_all[ilatsel,ilonsel] = helpers.confacc(np.round(all_lr_pred.squeeze())[1:],all_lr_true.squeeze()[1:],all_lr_conf[1:])
                    bs_lr_all[ilatsel,ilonsel] = helpers.brierscore(all_lr_pred.squeeze()[1:],all_lr_true.squeeze()[1:])
                
            if len(glob.glob(lr_pred_file))==0:
                with open(lr_pred_file,'wb') as f:
                    pickle.dump(lr_preds,f)

        self.lr_coeffs = lr_coeffs

        return lr_accuracy_all, bs_lr_all, lr_first_correct

    def cnn_evaluation(self,AllData):

        _, _, alltest = AllData.trainvaltest_recordmax_withrecordmax(self.trainvaltest,self.experiment_era,self.baselineera,self.inputlength,self.outputavgtime,-10,0) # need this to get the GMT vector
        inputtestGMT = alltest[1]

        bestaccs = np.empty((len(AllData.output_lat),len(AllData.output_lon),20)) + np.nan
        nulls = np.empty((len(AllData.output_lat),len(AllData.output_lon))) + np.nan
        nulls2 = np.empty((len(AllData.output_lat),len(AllData.output_lon))) + np.nan

        gmt_first_positive = np.empty((5,len(AllData.output_lat),len(AllData.output_lon)))+np.nan
        gmt_first_confident = np.empty((5,len(AllData.output_lat),len(AllData.output_lon)))+np.nan
        gmt_first_correct = np.empty((5,len(AllData.output_lat),len(AllData.output_lon)))+np.nan

        random_chance = np.empty((5,len(AllData.output_lat),len(AllData.output_lon)))+np.nan
        cnn_accuracy_ssp = np.empty((5,len(AllData.output_lat),len(AllData.output_lon),20))+np.nan

        cnn_bs = np.empty((len(AllData.output_lat),len(AllData.output_lon))) + np.nan

        regenerate = False

        for ilatsel,latsel in enumerate(AllData.output_lat):

            if (latsel>=0) & (regenerate==False):

                _, _, alltest = AllData.trainvaltest_recordmax_withrecordmax(self.trainvaltest,self.experiment_era,self.baselineera,self.inputlength,self.outputavgtime,10,0) # and then regenerate the GMT vector when we cross the equation
                inputtestGMT = alltest[1]

                regenerate = True
            
            fileopen = "predictions/"+self.modelfilefront+"avgtime_"+str(self.outputavgtime)+"_allssps_lat_"+str(latsel)+"_testing.pkl" # (nlon,nmodels=3,nsamples=3600)

            filecheck = glob.glob(fileopen)
            if len(filecheck) != 0:

                with open(filecheck[0],'rb') as f:
                    predictions = pickle.load(f)
                
                filetrue = "predictions/"+self.modelfilefront+"avgtime_"+str(self.outputavgtime)+"_allssps_lat_"+str(latsel)+"_truetesting.pkl"
                with open(filetrue,'rb') as f:
                    truths = pickle.load(f)
                
                avgtestpred = np.mean(predictions,axis=1)

                for ilonsel,lonsel in enumerate(AllData.output_lon):

                    if ~np.isnan(avgtestpred[ilonsel,0]):

                        filemetrics = "metrics/"+self.modelfilefront+"avgtime_"+str(self.outputavgtime)+"_allssps_lat_"+str(latsel)+"_lon_"+str(lonsel)+"_testing.json"
                        with open(filemetrics, 'r') as f:
                            testmetrics = json.load(f)

                        bestaccs[ilatsel,ilonsel,:] = testmetrics["test_accuracy"]
                        nulls[ilatsel,ilonsel] = np.max(testmetrics["test_class_imbalance"])

                        nulls2[ilatsel,ilonsel] = np.max([np.mean(truths[ilonsel]),1-np.mean(truths[ilonsel])])
                        cnn_bs[ilatsel,ilonsel] = helpers.brierscore(avgtestpred[ilonsel].squeeze(),truths[ilonsel].squeeze())

                        for issp,ssp in enumerate(self.ssplistplot):

                            inputGMTsel = helpers.grabssp(inputtestGMT,issp,AllData.lenhisttime,AllData.lenfuturetime,len(self.trainvaltest[-1]))
                            predsel = helpers.grabssp(avgtestpred[ilonsel],issp,AllData.lenhisttime,AllData.lenfuturetime,len(self.trainvaltest[-1]))
                            truesel = helpers.grabssp(truths[ilonsel],issp,AllData.lenhisttime,AllData.lenfuturetime,len(self.trainvaltest[-1]))

                            predclass = np.round(predsel)
                            predconf = np.where(predsel<0.5,1-predsel,predsel)

                            cnn_accuracy_ssp[issp,ilatsel,ilonsel] = helpers.confacc(predclass,truesel,predconf)

                            # first positive prediction
                            gmt_first_positive[issp,ilatsel,ilonsel] = helpers.firstpositive(predsel,inputGMTsel)

                            # first confident, positive prediction
                            gmt_first_confident[issp,ilatsel,ilonsel] = helpers.firstconfident(predsel,inputGMTsel)

                            # first correct, positive prediction 
                            gmt_first_correct[issp,ilatsel,ilonsel] = helpers.firstcorrect(predsel,truesel,inputGMTsel)

                            random_chance[issp,ilatsel,ilonsel] = np.max([np.mean(truesel),1-np.mean(truesel)])

        return bestaccs, cnn_bs, gmt_first_correct

    def obs_evaluation(self,AllObs):
        
        n_best = 3

        chopind = self.experiment_era_obs[0]-self.obstimerange[0]
        lenSH = len(AllObs.allGMT.squeeze())-self.outputavgtime-self.inputlength-chopind
        lenNH = len(AllObs.allGMT.squeeze())-self.outputavgtime-self.inputlength-chopind+1

        lr_obs_accuracy = np.empty((len(AllObs.output_lat),len(AllObs.output_lon)))+np.nan
        cnn_obs_accuracy = np.empty((len(AllObs.output_lat),len(AllObs.output_lon)))+np.nan

        cnn_obs_bs = np.empty((len(AllObs.output_lat),len(AllObs.output_lon)))+np.nan
        lr_obs_bs = np.empty((len(AllObs.output_lat),len(AllObs.output_lon)))+np.nan

        cnn_obs_firstpositive = np.empty((len(AllObs.output_lat),len(AllObs.output_lon)))+np.nan
        lr_obs_firstpositive = np.empty((len(AllObs.output_lat),len(AllObs.output_lon)))+np.nan

        cnn_tp = np.empty((len(AllObs.output_lat),len(AllObs.output_lon)))+np.nan
        cnn_fp = np.empty((len(AllObs.output_lat),len(AllObs.output_lon)))+np.nan

        lr_tp = np.empty((len(AllObs.output_lat),len(AllObs.output_lon)))+np.nan
        lr_fp = np.empty((len(AllObs.output_lat),len(AllObs.output_lon)))+np.nan

        cnn_precision_positiveclass = np.empty((len(AllObs.output_lat),len(AllObs.output_lon)))+np.nan
        lr_precision_positiveclass = np.empty((len(AllObs.output_lat),len(AllObs.output_lon)))+np.nan

        cnn_SH_pred = np.empty((int(len(AllObs.output_lat)/2),len(AllObs.output_lon),lenSH))+np.nan
        cnn_NH_pred = np.empty((int(len(AllObs.output_lat)/2),len(AllObs.output_lon),lenNH))+np.nan

        SH_true = np.empty((int(len(AllObs.output_lat)/2),len(AllObs.output_lon),lenSH))+np.nan
        NH_true = np.empty((int(len(AllObs.output_lat)/2),len(AllObs.output_lon),lenNH))+np.nan

        for ilat,lat in enumerate(AllObs.output_lat):
            for ilon,lon in enumerate(AllObs.output_lon):

                metricsout = "metrics/"+self.modelfilefront+"avgtime_"+str(self.outputavgtime)+"_allssps_lat_"+str(lat)+"_lon_"+str(lon)+"_seed*.json"
                filelist = glob.glob(metricsout)

                if len(filelist)!=0:
                    bestseeds = helpers.get_best_files(filelist,n_best)

                    obsinput,obsinputgmt,obsinputpriorrecord,obsoutput = AllObs.obs_recordmax_withrecordmax(self.experiment_era_obs,self.baselineera_obs,self.inputlength,self.outputavgtime,lat,lon)

                    obsinput_t = torch.tensor(obsinput,dtype=torch.float32)
                    obsinputgmt_t = torch.tensor(obsinputgmt,dtype=torch.float32)
                    obsinputpriorrecord_t = torch.tensor(obsinputpriorrecord,dtype=torch.float32)
                    obsinputvectors_t = torch.cat((obsinputgmt_t,obsinputpriorrecord_t),axis=-1)

                    obspredall = np.empty((3,len(obsinputgmt)))

                    if ~np.isnan(AllObs.outputsummer[0]):

                        for iseed,seed in enumerate(bestseeds):
                            # load the model

                            loadfile = "models/"+self.modelfilefront+"avgtime_"+str(self.outputavgtime)+"_allssps_lat_"+str(lat)+"_lon_"+str(lon)+"_seed_"+str(seed)+".pt"
                            cnn = buildmodel.CNNclassifier(obsinput_t, obsinputvectors_t, 2).to('cpu')
                            cnn.load_state_dict(torch.load(loadfile,map_location=torch.device('cpu'), weights_only=False))
                            cnn.to(device)

                            with torch.no_grad():
                                cnn.eval()
                                obspred = cnn(obsinput_t.to(device), obsinputvectors_t.to(device)).cpu().numpy()

                            obspredall[iseed] = obspred[:,1]

                        obspredavg = np.mean(obspredall,axis=0)
                        obspredclass = np.round(obspredavg)
                        if lat<0:
                            cnn_SH_pred[ilat,ilon,:] = obspredavg
                            SH_true[ilat,ilon,:] = obsoutput
                        else:
                            cnn_NH_pred[ilat-9,ilon,:] = obspredavg
                            NH_true[ilat-9,ilon,:] = obsoutput

                        cnn_obs_accuracy[ilat,ilon] = np.mean(obsoutput==obspredclass)
                        cnn_obs_bs[ilat,ilon] = helpers.brierscore(obspredavg.squeeze(),obsoutput.squeeze())
                        cnn_obs_firstpositive[ilat,ilon] = helpers.firstpositive(obspredavg,obsinputgmt)

                        cnn_tp[ilat,ilon] = np.sum(1*((obspredclass==obsoutput)&(obspredclass==1)))
                        cnn_fp[ilat,ilon] = np.sum(1*((obspredclass!=obsoutput)&(obspredclass==1)))

                        cnn_precision_positiveclass[ilat,ilon] = np.sum(1*((obspredclass==obsoutput)&(obspredclass==1)))/np.sum(1*(obspredclass==1))

                        lrmodelload = "lr_models/"+self.modelfilefront + "LogisticRegression_avgtime_"+str(self.outputavgtime)+"_lat_"+str(lat)+"_lon_"+str(lon)+".joblib"
                        lrmodel = joblib.load(lrmodelload)

                        lrpred = lrmodel.predict_proba(obsinputgmt)[:,1]
                        lrpredclass = np.round(lrpred)

                        lr_obs_bs[ilat,ilon] = helpers.brierscore(lrpred,obsoutput.squeeze())
                        lr_obs_accuracy[ilat,ilon] = np.mean(lrpredclass.squeeze()==obsoutput.squeeze())
                        lr_obs_firstpositive[ilat,ilon] = helpers.firstpositive(lrpred,obsinputgmt)

                        lr_tp[ilat,ilon] = np.sum(1*((lrpredclass==obsoutput)&(lrpredclass==1)))
                        lr_fp[ilat,ilon] = np.sum(1*((lrpredclass!=obsoutput)&(lrpredclass==1)))

                        lr_precision_positiveclass[ilat,ilon] = np.sum(1*((lrpredclass==obsoutput)&(lrpredclass==1)))/np.sum(1*(lrpredclass==1))

        self.cnn_tp = cnn_tp
        self.cnn_fp = cnn_fp

        self.lr_tp = lr_tp
        self.lr_fp = lr_fp

        self.lr_precision = lr_precision_positiveclass
        self.cnn_precision = cnn_precision_positiveclass

        self.cnn_SH_pred = cnn_SH_pred
        self.cnn_NH_pred = cnn_NH_pred

        self.NH_true = NH_true
        self.SH_true = SH_true

        return [cnn_obs_accuracy, cnn_obs_bs, cnn_obs_firstpositive],[lr_obs_accuracy, lr_obs_bs, lr_obs_firstpositive]


    # def moreobsthings(self,AllObs):

    #     lr_obs_accuracy = np.empty((len(AllObs.output_lat),len(AllObs.output_lon)))+np.nan
    #     cnn_obs_accuracy = np.empty((len(AllObs.output_lat),len(AllObs.output_lon)))+np.nan

    #     for ilat,lat in enumerate(AllObs.output_lat):
    #         for ilon,lon in enumerate(AllObs.output_lon):

    #             metricsout = "metrics/"+self.modelfilefront+"avgtime_"+str(self.outputavgtime)+"_allssps_lat_"+str(lat)+"_lon_"+str(lon)+"_seed*.json"
    #             filelist = glob.glob(metricsout)

    #             if len(filelist)!=0:
    #                 bestseeds = helpers.get_best_files(filelist,n_best)

    #                 obsinput,obsinputgmt,obsinputpriorrecord,obsoutput = AllObs.obs_recordmax_withrecordmax(self.experiment_era_obs,self.baselineera_obs,self.inputlength,self.outputavgtime,lat,lon)

    #                 obsinput_t = torch.tensor(obsinput,dtype=torch.float32)
    #                 obsinputgmt_t = torch.tensor(obsinputgmt,dtype=torch.float32)
    #                 obsinputpriorrecord_t = torch.tensor(obsinputpriorrecord,dtype=torch.float32)
    #                 obsinputvectors_t = torch.cat((obsinputgmt_t,obsinputpriorrecord_t),axis=-1)

    #                 obspredall = np.empty((3,len(obsinputgmt)))



    def GridPointSel(self,AllData,latsel,lonsel):
        
        """For a particular gridpoint, get the input and output data, output predictions, three best CNNs, and baseline
        logistic regression models
        AllData: the DataHolder class with the data
        latsel: selected latitude
        lonsel: selected longitude

        outputs
        inputSSTtest: SSTs input to CNN
        inputGMTtest: GMT input to CNN
        inputpriormaxtest: prior max record input to CNN
        outputtest: ground truth outputs for CNN
        inputssptest: flags indicating whether data comes from hist or sspx-x.x
        cnns: list of 3 best cnns
        lr_model: logistic regression model
        cnn_pred: predictions from the three best cnns (confidence in positive class prediction)
        """

        valmetricsout = "metrics/"+self.modelfilefront+"avgtime_"+str(self.outputavgtime)+"_allssps_lat_"+str(latsel)+"_lon_"+str(lonsel)+"_seed*.json"
        filelist = glob.glob(valmetricsout)
        ilonsel = int(lonsel/10) # oops all hard coded

        if len(filelist)==0:
            print('model does not exist')

        else:
            _, _, alltest = AllData.trainvaltest_recordmax_withrecordmax(self.trainvaltest,self.experiment_era,self.baselineera,self.inputlength,self.outputavgtime,latsel,lonsel)
            _,_,self.inputssptest = AllData.trainvaltest_recordmax_ssps(self.trainvaltest,self.experiment_era,self.inputlength,self.outputavgtime,latsel)

            self.inputSSTtest = alltest[0]
            self.inputGMTtest = alltest[1]
            self.inputpriormaxtest = alltest[2]
            self.outputtest = alltest[3]

            n_best = 3 # oops all hard coded
            bestseeds = helpers.get_best_files(filelist,n_best)
            
            cnns = []
            for iseed,seed in enumerate(bestseeds):
                # load the model

                loadfile = "models/"+self.modelfilefront+"avgtime_"+str(self.outputavgtime)+"_allssps_lat_"+str(latsel)+"_lon_"+str(lonsel)+"_seed_"+str(seed)+".pt"
                cnn = buildmodel.CNNclassifier(torch.tensor(self.inputSSTtest), torch.cat((torch.tensor(self.inputGMTtest),torch.tensor(self.inputpriormaxtest)),axis=1), 2).to('cpu')
                cnn.load_state_dict(torch.load(loadfile,map_location=torch.device('cpu'), weights_only=False))
                cnns.append(cnn)

            self.cnns = cnns

            lrmodelload = "lr_models/"+self.modelfilefront + "LogisticRegression_avgtime_"+str(self.outputavgtime)+"_lat_"+str(latsel)+"_lon_"+str(lonsel)+".joblib"
            self.lrmodel = joblib.load(lrmodelload)

            predfile = "predictions/"+self.modelfilefront+"avgtime_"+str(self.outputavgtime)+"_allssps_lat_"+str(latsel)+"_testing.pkl" # (nlon,nmodels=3,nsamples=3600)

            with open(predfile,'rb') as f:
                predictions = pickle.load(f)
            
            self.cnnpred = predictions[ilonsel] # 2 models x ntestsample predictions



class HistoricMax:
    def __init__(self,analysisparams):
        
        self.modelfilefront = "MPI_histrecord_"

        self.trainvaltest = analysisparams["trainvaltest"]
        self.experiment_era = analysisparams["experiment_era"]
        self.baselineera = analysisparams["baselineera"]
        self.inputlength = analysisparams["inputlength"]
        self.outputavgtime = analysisparams["outputavgtime"]
        self.ssplist = analysisparams["ssps"]
        self.ssplistplot = ["hist"]+ self.ssplist 
        self.experiment_era_obs = analysisparams["experiment_era_obs"]
        self.baselineera_obs = analysisparams["baselineera_obs"]

    def logistic_regression(self,AllData,sspkeep):
        
        alltrain, _, alltest = AllData.trainvaltest_histrecordmax(self.trainvaltest,self.experiment_era,self.baselineera,self.inputlength,self.outputavgtime,-20,0)

        inputtrainGMT = alltrain[1]
        inputtestGMT = alltest[1]

        inputssptrain,_,inputssptest = AllData.trainvaltest_recordmax_ssps(self.trainvaltest,self.experiment_era,self.inputlength,self.outputavgtime,-20)

        inputssptrain = np.round((inputssptrain.squeeze()+0.2)*10)
        inputssptest = np.round((inputssptest.squeeze()+0.2)*10) # EMILY STOP COMMENTING THIS OUT IT IS NECESSARY YOU IDIOT

        regenerate = False

        lr_first_positive = np.empty((5,len(AllData.output_lat),len(AllData.output_lon)))+np.nan
        lr_first_correct = np.empty((5,len(AllData.output_lat),len(AllData.output_lon)))+np.nan
        lr_first_confident = np.empty((5,len(AllData.output_lat),len(AllData.output_lon)))+np.nan

        lr_accuracy_ssp = np.empty((5,len(AllData.output_lat),len(AllData.output_lon),20))+np.nan
        bs_lr_ssp = np.empty((5,len(AllData.output_lat),len(AllData.output_lon)))+np.nan

        lr_accuracy_all = np.empty((len(AllData.output_lat),len(AllData.output_lon),20))+np.nan
        bs_lr_all = np.empty((len(AllData.output_lat),len(AllData.output_lon)))+np.nan

        lr_coeffs = np.empty((len(AllData.output_lat),len(AllData.output_lon)))+np.nan

        for ilatsel,latsel in enumerate(AllData.output_lat):
            print(latsel)

            if (latsel>=0) & (regenerate==False):

                alltrain, _, alltest = AllData.trainvaltest_histrecordmax(self.trainvaltest,self.experiment_era,self.baselineera,self.inputlength,self.outputavgtime,latsel,0)
                inputssptrain,_,inputssptest = AllData.trainvaltest_recordmax_ssps(self.trainvaltest,self.experiment_era,self.inputlength,self.outputavgtime,latsel)

                inputtrainGMT = alltrain[1]
                inputtestGMT = alltest[1]   

                inputssptrain,_,inputssptest = AllData.trainvaltest_recordmax_ssps(self.trainvaltest,self.experiment_era,self.inputlength,self.outputavgtime,0)

                inputssptrain = np.round((inputssptrain.squeeze()+0.2)*10)
                inputssptest = np.round((inputssptest.squeeze()+0.2)*10)

                regenerate = True

            lr_pred_file = "predictions/"+self.modelfilefront+"avgtime_"+str(self.outputavgtime)+"_allssps_lat_"+str(latsel)+"_logisticregression.pkl"
            if len(glob.glob(lr_pred_file))==0:
                lr_preds = np.empty((2,36,len(inputtestGMT.squeeze())))

            for ilonsel,lonsel in enumerate(AllData.output_lon):

                outputtrainall, _, outputtestall = AllData.trainvaltest_histrecordmax_outputonly(self.trainvaltest,self.experiment_era,self.baselineera,self.inputlength,self.outputavgtime,latsel,lonsel)
                
                if (np.mean(outputtrainall)!=0) & (np.mean(outputtrainall)!=1):

                    #grab hist only
                    inputtrainGMT = inputtrainGMT[:,[0]]
                    inputtestGMT = inputtestGMT[:,[0]]

                    histtraingmt = inputtrainGMT[inputssptrain==0]
                    histtestgmt = inputtestGMT[inputssptest==0]

                    outputtrainhist = outputtrainall[inputssptrain==0]
                    outputtesthist = outputtestall[inputssptest==0]

                    ssplabelhist = inputssptest[inputssptest==0]

                    all_lr_pred = np.empty(1)
                    all_lr_conf = np.empty(1)
                    all_lr_true = np.empty(1)

                    for issp in range(1,5):

                        inputtraingmtssp = inputtrainGMT[inputssptrain==issp]
                        allinputtraingmt = np.concatenate((histtraingmt,inputtraingmtssp),axis=0)

                        outputssptrain = outputtrainall[inputssptrain==issp]
                        alloutputtrain = np.concatenate((outputtrainhist,outputssptrain),axis=0)

                        inputtestgmtssp = inputtestGMT[inputssptest==issp]
                        allinputtestgmt = np.concatenate((histtestgmt,inputtestgmtssp),axis=0)

                        outputssptest = outputtestall[inputssptest==issp]
                        alloutputtest = np.concatenate((outputtesthist,outputssptest),axis=0)

                        ssplabelssp = inputssptest[inputssptest==issp]
                        ssplabelsall = np.concatenate((ssplabelhist,ssplabelssp),axis=0)

                        # get the CNN predictions

                        model = LogisticRegression(random_state=42)

                        # Fit the model
                        model.fit(allinputtraingmt, alloutputtrain.squeeze())

                        y_pred_all = model.predict(allinputtestgmt)
                        y_pred_all_prob = model.predict_proba(allinputtestgmt)
                        y_conf_all = np.max(y_pred_all_prob,axis=1)

                        lr_accuracy_ssp[issp,ilatsel,ilonsel] = helpers.confacc(y_pred_all.squeeze(),alloutputtest.squeeze(),y_conf_all)

                        lr_first_positive[issp,ilatsel,ilonsel] = helpers.firstpositive(y_pred_all_prob[:,1],allinputtestgmt.squeeze())
                        lr_first_correct[issp,ilatsel,ilonsel] = helpers.firstcorrect(y_pred_all_prob[:,1],alloutputtest.squeeze(),allinputtestgmt.squeeze())
                        lr_first_confident[issp,ilatsel,ilonsel] = helpers.firstconfident(y_pred_all_prob[:,1],allinputtestgmt.squeeze())

                        # save predictions of all ssps and only hist for selected ssp

                        all_lr_pred = np.append(all_lr_pred,y_pred_all_prob[ssplabelsall==issp,1])
                        all_lr_conf = np.append(all_lr_conf,np.max(y_pred_all_prob[ssplabelsall==issp,:],axis=1))
                        all_lr_true = np.append(all_lr_true,outputssptest)

                        if issp == sspkeep:

                            lrmodelsavefile = "lr_models/"+ self.modelfilefront + "LogisticRegression_avgtime_"+str(self.outputavgtime)+"_lat_"+str(latsel)+"_lon_"+str(lonsel)+".joblib"
                            joblib.dump(model, lrmodelsavefile)

                            all_lr_pred = np.append(all_lr_pred,y_pred_all_prob[ssplabelsall==0,1])
                            all_lr_conf = np.append(all_lr_conf,np.max(y_pred_all_prob[ssplabelsall==0,:],axis=1))
                            all_lr_true = np.append(all_lr_true,outputtesthist)

                            y_pred_hist_prob = model.predict_proba(histtestgmt)
                            lr_first_correct[0,ilatsel,ilonsel] = helpers.firstcorrect(y_pred_hist_prob[:,1],outputtesthist.squeeze(),histtestgmt.squeeze())

                            lr_coeffs[ilatsel,ilonsel] = model.coef_[0]

                    if len(glob.glob(lr_pred_file))==0:
                        lr_preds[0,ilonsel] = all_lr_pred.squeeze()[1:]
                        lr_preds[1,ilonsel] = all_lr_true.squeeze()[1:]

                    lr_accuracy_all[ilatsel,ilonsel] = helpers.confacc(np.round(all_lr_pred.squeeze())[1:],all_lr_true.squeeze()[1:],all_lr_conf[1:])
                    bs_lr_all[ilatsel,ilonsel] = helpers.brierscore(all_lr_pred.squeeze()[1:],all_lr_true.squeeze()[1:])

            if len(glob.glob(lr_pred_file))==0:
                with open(lr_pred_file,'wb') as f:
                    pickle.dump(lr_preds,f)

        self.lr_coeffs = lr_coeffs

        return lr_accuracy_all, bs_lr_all, lr_first_correct

    def cnn_evaluation(self,AllData):

        _, _, alltest = AllData.trainvaltest_histrecordmax(self.trainvaltest,self.experiment_era,self.baselineera,self.inputlength,self.outputavgtime,-10,0) # need this to get the GMT vector
        inputtestGMT = alltest[1]
        # _, inputtestGMT, _ = DataHolder.tensortime_onehot_withrecordmax(alltest,nclasses=2) # gmt vector
        # inputtestGMT = inputtestGMT[:,[0]].numpy()

        bestaccs = np.empty((len(AllData.output_lat),len(AllData.output_lon),20)) + np.nan
        nulls = np.empty((len(AllData.output_lat),len(AllData.output_lon))) + np.nan
        nulls2 = np.empty((len(AllData.output_lat),len(AllData.output_lon))) + np.nan

        gmt_first_positive = np.empty((5,len(AllData.output_lat),len(AllData.output_lon)))+np.nan
        gmt_first_confident = np.empty((5,len(AllData.output_lat),len(AllData.output_lon)))+np.nan
        gmt_first_correct = np.empty((5,len(AllData.output_lat),len(AllData.output_lon)))+np.nan

        random_chance = np.empty((5,len(AllData.output_lat),len(AllData.output_lon)))+np.nan
        cnn_accuracy_ssp = np.empty((5,len(AllData.output_lat),len(AllData.output_lon),20))+np.nan

        cnn_bs = np.empty((len(AllData.output_lat),len(AllData.output_lon))) + np.nan

        regenerate = False

        for ilatsel,latsel in enumerate(AllData.output_lat):

            if (latsel>=0) & (regenerate==False):

                _, _, alltest = AllData.trainvaltest_recordmax_withrecordmax(self.trainvaltest,self.experiment_era,self.baselineera,self.inputlength,self.outputavgtime,10,0) # and then regenerate the GMT vector when we cross the equation
                inputtestGMT = alltest[1]
                # _, inputtestGMT, _ = DataHolder.tensortime_onehot_withrecordmax(alltest,nclasses=2) 
                # inputtestGMT = inputtestGMT[:,[0]].numpy()

                regenerate = True
            
            fileopen = "predictions/"+self.modelfilefront+"avgtime_"+str(self.outputavgtime)+"_allssps_lat_"+str(latsel)+"_testing.pkl" # (nlon,nmodels=3,nsamples=3600)

            filecheck = glob.glob(fileopen)
            if len(filecheck) != 0:

                with open(filecheck[0],'rb') as f:
                    predictions = pickle.load(f)
                
                filetrue = "predictions/"+self.modelfilefront+"avgtime_"+str(self.outputavgtime)+"_allssps_lat_"+str(latsel)+"_truetesting.pkl"
                with open(filetrue,'rb') as f:
                    truths = pickle.load(f)
                
                avgtestpred = np.mean(predictions,axis=1)

                for ilonsel,lonsel in enumerate(AllData.output_lon):

                    if ~np.isnan(avgtestpred[ilonsel,0]):

                        filemetrics = "metrics/"+self.modelfilefront+"avgtime_"+str(self.outputavgtime)+"_allssps_lat_"+str(latsel)+"_lon_"+str(lonsel)+"_testing.json"
                        with open(filemetrics, 'r') as f:
                            testmetrics = json.load(f)

                        bestaccs[ilatsel,ilonsel,:] = testmetrics["test_accuracy"]
                        nulls[ilatsel,ilonsel] = np.max(testmetrics["test_class_imbalance"])

                        nulls2[ilatsel,ilonsel] = np.max([np.mean(truths[ilonsel]),1-np.mean(truths[ilonsel])])
                        cnn_bs[ilatsel,ilonsel] = helpers.brierscore(avgtestpred[ilonsel].squeeze(),truths[ilonsel].squeeze())

                        for issp,ssp in enumerate(self.ssplistplot):

                            inputGMTsel = helpers.grabssp(inputtestGMT,issp,AllData.lenhisttime,AllData.lenfuturetime,len(self.trainvaltest[-1]))
                            predsel = helpers.grabssp(avgtestpred[ilonsel],issp,AllData.lenhisttime,AllData.lenfuturetime,len(self.trainvaltest[-1]))
                            truesel = helpers.grabssp(truths[ilonsel],issp,AllData.lenhisttime,AllData.lenfuturetime,len(self.trainvaltest[-1]))

                            predclass = np.round(predsel)
                            predconf = np.where(predsel<0.5,1-predsel,predsel)

                            cnn_accuracy_ssp[issp,ilatsel,ilonsel] = helpers.confacc(predclass,truesel,predconf)

                            # first positive prediction
                            gmt_first_positive[issp,ilatsel,ilonsel] = helpers.firstpositive(predsel,inputGMTsel)

                            # first confident, positive prediction
                            gmt_first_confident[issp,ilatsel,ilonsel] = helpers.firstconfident(predsel,inputGMTsel)

                            # first correct, positive prediction 
                            gmt_first_correct[issp,ilatsel,ilonsel] = helpers.firstcorrect(predsel,truesel,inputGMTsel)

                            random_chance[issp,ilatsel,ilonsel] = np.max([np.mean(truesel),1-np.mean(truesel)])

        return bestaccs, cnn_bs, gmt_first_correct

    def obs_evaluation(self,AllObs):
        
        n_best = 3
        lr_obs_accuracy = np.empty((len(AllObs.output_lat),len(AllObs.output_lon)))+np.nan
        cnn_obs_accuracy = np.empty((len(AllObs.output_lat),len(AllObs.output_lon)))+np.nan

        cnn_obs_bs = np.empty((len(AllObs.output_lat),len(AllObs.output_lon)))+np.nan
        lr_obs_bs = np.empty((len(AllObs.output_lat),len(AllObs.output_lon)))+np.nan

        cnn_obs_firstpositive = np.empty((len(AllObs.output_lat),len(AllObs.output_lon)))+np.nan
        lr_obs_firstpositive = np.empty((len(AllObs.output_lat),len(AllObs.output_lon)))+np.nan

        cnn_tp = np.empty((len(AllObs.output_lat),len(AllObs.output_lon)))+np.nan
        cnn_fp = np.empty((len(AllObs.output_lat),len(AllObs.output_lon)))+np.nan

        lr_tp = np.empty((len(AllObs.output_lat),len(AllObs.output_lon)))+np.nan
        lr_fp = np.empty((len(AllObs.output_lat),len(AllObs.output_lon)))+np.nan

        cnn_precision_positiveclass = np.empty((len(AllObs.output_lat),len(AllObs.output_lon)))+np.nan
        lr_precision_positiveclass = np.empty((len(AllObs.output_lat),len(AllObs.output_lon)))+np.nan

        for ilat,lat in enumerate(AllObs.output_lat):
            for ilon,lon in enumerate(AllObs.output_lon):

                metricsout = "metrics/"+self.modelfilefront+"avgtime_"+str(self.outputavgtime)+"_allssps_lat_"+str(lat)+"_lon_"+str(lon)+"_seed*.json"
                filelist = glob.glob(metricsout)

                if len(filelist)!=0:
                    bestseeds = helpers.get_best_files(filelist,n_best)

                    obsinput,obsinputgmt,obsinputpriorrecord,obsoutput = AllObs.obs_histrecordmax(self.experiment_era_obs,self.baselineera_obs,self.inputlength,self.outputavgtime,lat,lon)

                    obsinput_t = torch.tensor(obsinput,dtype=torch.float32)
                    obsinputgmt_t = torch.tensor(obsinputgmt,dtype=torch.float32)
                    obsinputpriorrecord_t = torch.tensor(obsinputpriorrecord,dtype=torch.float32)
                    obsinputvectors_t = torch.cat((obsinputgmt_t,obsinputpriorrecord_t),axis=-1)

                    obspredall = np.empty((3,len(obsinputgmt)))

                    if ~np.isnan(AllObs.outputsummer[0]):

                        for iseed,seed in enumerate(bestseeds):
                            # load the model

                            loadfile = "models/"+self.modelfilefront+"avgtime_"+str(self.outputavgtime)+"_allssps_lat_"+str(lat)+"_lon_"+str(lon)+"_seed_"+str(seed)+".pt"
                            cnn = buildmodel.CNNclassifier(obsinput_t, obsinputvectors_t, 2).to('cpu')
                            cnn.load_state_dict(torch.load(loadfile,map_location=torch.device('cpu'), weights_only=False))
                            cnn.to(device)

                            with torch.no_grad():
                                cnn.eval()
                                obspred = cnn(obsinput_t.to(device), obsinputvectors_t.to(device)).cpu().numpy()

                            obspredall[iseed] = obspred[:,1]

                        obspredavg = np.mean(obspredall,axis=0)
                        obspredclass = np.round(obspredavg)

                        cnn_tp[ilat,ilon] = np.sum(1*((obspredclass==obsoutput)&(obspredclass==1)))
                        cnn_fp[ilat,ilon] = np.sum(1*((obspredclass!=obsoutput)&(obspredclass==1)))

                        cnn_obs_accuracy[ilat,ilon] = np.mean(obsoutput==obspredclass)
                        cnn_obs_bs[ilat,ilon] = helpers.brierscore(obspredavg.squeeze(),obsoutput.squeeze())
                        cnn_obs_firstpositive[ilat,ilon] = helpers.firstpositive(obspredavg,obsinputgmt)
                        
                        cnn_precision_positiveclass[ilat,ilon] = np.sum(1*((obspredclass==obsoutput)&(obspredclass==1)))/np.sum(1*(obspredclass==1))

                        lrmodelload = "lr_models/"+self.modelfilefront + "LogisticRegression_avgtime_"+str(self.outputavgtime)+"_lat_"+str(lat)+"_lon_"+str(lon)+".joblib"
                        lrmodel = joblib.load(lrmodelload)

                        lrpred = lrmodel.predict_proba(obsinputgmt)[:,1]
                        lrpredclass = np.round(lrpred)

                        lr_obs_bs[ilat,ilon] = helpers.brierscore(lrpred,obsoutput.squeeze())
                        lr_obs_accuracy[ilat,ilon] = np.mean(lrpredclass.squeeze()==obsoutput.squeeze())
                        lr_obs_firstpositive[ilat,ilon] = helpers.firstpositive(lrpred,obsinputgmt)

                        lr_tp[ilat,ilon] = np.sum(1*((lrpredclass==obsoutput)&(lrpredclass==1)))
                        lr_fp[ilat,ilon] = np.sum(1*((lrpredclass!=obsoutput)&(lrpredclass==1)))

                        lr_precision_positiveclass[ilat,ilon] = np.sum(1*((lrpredclass==obsoutput)&(lrpredclass==1)))/np.sum(1*(lrpredclass==1))

        self.cnn_tp = cnn_tp
        self.cnn_fp = cnn_fp

        self.lr_tp = lr_tp
        self.lr_fp = lr_fp

        self.cnn_precision = cnn_precision_positiveclass
        self.lr_precision = lr_precision_positiveclass

        return [cnn_obs_accuracy, cnn_obs_bs, cnn_obs_firstpositive],[lr_obs_accuracy, lr_obs_bs, lr_obs_firstpositive]

    def GridPointSel(self,AllData,latsel,lonsel):
        
        """For a particular gridpoint, get the input and output data, output predictions, three best CNNs, and baseline
        logistic regression models
        AllData: the DataHolder class with the data
        latsel: selected latitude
        lonsel: selected longitude

        outputs
        inputSSTtest: SSTs input to CNN
        inputGMTtest: GMT input to CNN
        inputpriormaxtest: prior max record input to CNN
        outputtest: ground truth outputs for CNN
        inputssptest: flags indicating whether data comes from hist or sspx-x.x
        cnns: list of 3 best cnns
        lr_model: logistic regression model
        cnn_pred: predictions from the three best cnns (confidence in positive class prediction)
        """

        valmetricsout = "metrics/"+self.modelfilefront+"avgtime_"+str(self.outputavgtime)+"_allssps_lat_"+str(latsel)+"_lon_"+str(lonsel)+"_seed*.json"
        filelist = glob.glob(valmetricsout)
        ilonsel = int(lonsel/10) # oops all hard coded

        if len(filelist)==0:
            print('model does not exist')

        else:
            _, _, alltest = AllData.trainvaltest_histrecordmax(self.trainvaltest,self.experiment_era,self.baselineera,self.inputlength,self.outputavgtime,latsel,lonsel)
            _,_,self.inputssptest = AllData.trainvaltest_recordmax_ssps(self.trainvaltest,self.experiment_era,self.inputlength,self.outputavgtime,latsel)

            self.inputSSTtest = alltest[0]
            self.inputGMTtest = alltest[1]
            self.inputpriormaxtest = alltest[2]
            self.outputtest = alltest[3]

            n_best = 3 # oops all hard coded
            bestseeds = helpers.get_best_files(filelist,n_best)
            
            cnns = []
            for iseed,seed in enumerate(bestseeds):
                # load the model

                loadfile = "models/"+self.modelfilefront+"avgtime_"+str(self.outputavgtime)+"_allssps_lat_"+str(latsel)+"_lon_"+str(lonsel)+"_seed_"+str(seed)+".pt"
                cnn = buildmodel.CNNclassifier(torch.tensor(self.inputSSTtest), torch.cat((torch.tensor(self.inputGMTtest),torch.tensor(self.inputpriormaxtest)),axis=1), 2).to('cpu')
                cnn.load_state_dict(torch.load(loadfile,map_location=torch.device('cpu'), weights_only=False))
                cnns.append(cnn)

            self.cnns = cnns

            lrmodelload = "lr_models/"+self.modelfilefront + "LogisticRegression_avgtime_"+str(self.outputavgtime)+"_lat_"+str(latsel)+"_lon_"+str(lonsel)+".joblib"
            self.lrmodel = joblib.load(lrmodelload)

            predfile = "predictions/"+self.modelfilefront+"avgtime_"+str(self.outputavgtime)+"_allssps_lat_"+str(latsel)+"_testing.pkl" # (nlon,nmodels=3,nsamples=3600)

            with open(predfile,'rb') as f:
                predictions = pickle.load(f)
            
            self.cnnpred = predictions[ilonsel] # 2 models x ntestsample predictions

def BlockBootStrap(analysisparams,latvec,lonvec,modelfilefront,nboots):
        
    outputavgtime = analysisparams["outputavgtime"]

    bs_cnn_boots = np.empty((len(latvec),len(lonvec),nboots))+np.nan
    bs_lr_boots = np.empty((len(latvec),len(lonvec),nboots))+np.nan

    for ilatsel,latsel in enumerate(latvec):

        if latsel < -70:
            print('oops no south pole')
        else:

            cnnpredfileopen = "predictions/"+modelfilefront+"avgtime_"+str(outputavgtime)+"_allssps_lat_"+str(latsel)+"_testing.pkl" # (nlon,nmodels=3,nsamples=3600ish)
            truepredfileopen = "predictions/"+modelfilefront+"avgtime_"+str(outputavgtime)+"_allssps_lat_"+str(latsel)+"_truetesting.pkl" # (nlon,nsamples=3600ish)
            lrpredfileopen = "predictions/"+modelfilefront+"avgtime_"+str(outputavgtime)+"_allssps_lat_"+str(latsel)+"_logisticregression.pkl" #((pred,true),nlon,nsamples=3600ish)

            with open(cnnpredfileopen,'rb') as f:
                cnnpred = pickle.load(f)
                cnnpred = np.mean(cnnpred,axis=1)            

            with open(truepredfileopen,'rb') as f:
                cnntrue = pickle.load(f)                        

            with open(lrpredfileopen,'rb') as f:
                lrpredtrue = pickle.load(f)
                lrpred = lrpredtrue[0]
                lrtrue = lrpredtrue[1]

            bs_cnn_boots[ilatsel,:,:], bs_lr_boots[ilatsel,:,:] = compare_models(cnnpred,lrpred,cnntrue,lrtrue,outputavgtime,nboots)

    return bs_cnn_boots,bs_lr_boots


def compare_models(preds_cnn,preds_lr,truth_cnn,truth_lr,leadtime,nboots):

    """
    Compare CNN to logistic regression using a block bootstrap of length leadtime
        preds_cnn: CNN predictions, longitude x sample 
        preds_lr: logistic regresion predictions, longitude x sample 
        truth: grountruths for CNN, longitude x sample 
        leadtime: outputavgtime for the predictions
        nboots: number of time to calculate the bootstrap
        """

    nsamps = preds_cnn.shape[-1] # total number of samples
    samplelength = int(preds_lr.shape[-1]/leadtime) # number to grab in bootstrap

    bs_cnn_boots = np.empty((36,nboots)) # longitude x bootnumber
    bs_lr_boots = np.empty((36,nboots)) # longitude x boot number

    for iboot in range(nboots):

        samplesel = np.random.choice(nsamps-leadtime,samplelength,replace=True)
        samplemat = samplesel[:,np.newaxis]+np.arange(leadtime)
        allsamps = samplemat.flatten()

        bs_cnn_boots[:,iboot] = np.mean(np.square(preds_cnn[:,allsamps]-truth_cnn[:,allsamps]),axis=-1)
        bs_lr_boots[:,iboot] = np.mean(np.square(preds_lr[:,allsamps]-truth_lr[:,allsamps]),axis=-1)
    
    return bs_cnn_boots,bs_lr_boots

def nino34(analysisparams,AllObs):

    inputlength = analysisparams["inputlength"]
    outputavgtime = analysisparams["outputavgtime"]
    experiment_era_obs = analysisparams["experiment_era_obs"]
    obstimerange = analysisparams["obstimerange"]

    obssst = AllObs.allinput
    obslon = np.arange(2,362,4)
    obslat = np.arange(-88,92,4)

    obs34reg = obssst[:,(obslat>-5)&(obslat<5),]
    obs34reg = obs34reg[:,:,(obslon>190)&(obslon<240)]

    nino34 = np.mean(obs34reg,axis=(1,2))

    cut1 = experiment_era_obs[0]-obstimerange[0]

    nino34out = nino34[cut1+inputlength:-1*outputavgtime]

    return nino34out

def grabsst(analysisparams,AllObs):

    experiment_era_obs = analysisparams["experiment_era_obs"]
    obstimerange = analysisparams["obstimerange"]

    obssst = AllObs.allinput
    cut1 = experiment_era_obs[0]-obstimerange[0]

    obssst_cut = obssst[cut1:]

    return obssst_cut

def grabt2m(analysisparams,AllObs):

    experiment_era_obs = analysisparams["experiment_era_obs"]
    obstimerange = analysisparams["obstimerange"]

    obst2m = AllObs.alloutput
    cut1 = experiment_era_obs[0]-obstimerange[0]

    obst2m_cut = obst2m[cut1:]

    return obst2m_cut