#File for preprocessing GCM large ensembles

import xarray as xr
import xesmf as xe
import numpy as np

import glob
import pickle
import time 

#%%

eravardict = {"tas":"t2m",
              "tos":"sst"}

class GriddedMPISummer:
    def __init__(self,params):

        outres = params["outres"]
        ssp = params["ssp"] 
        timerange = params["timerange"]
        filefront = params["filefront"]
        var = params["outputvar"]

        # fileout = "data/"+ var + "MPI_regridded_summertime_"+str(outres)+"x"+str(outres)+".pkl"
        fileout = "data/" + filefront +"summertime_" + ssp + "_" + var + "_"+str(timerange[0])+"-"+str(timerange[1])+".pkl"
        filecheck = glob.glob(fileout)

        if len(filecheck)==0:

            time1 = time.time()

            alldata_in = pulldata(var,ssp)

            allseasonal = alldata_in.rolling(time=3).mean() # seasonal

            allseasonal = allseasonal.groupby("time.month")

            all_a_summer = allseasonal[2]
            all_b_summer = allseasonal[8]

            latvec = np.arange(-90,90,10)
            lonvec = np.arange(0,360,10)

            all_a_summer = all_a_summer.sel(time=slice(str(timerange[0]),str(timerange[1])))
            all_b_summer = all_b_summer.sel(time=slice(str(timerange[0]),str(timerange[1])))

            indims = all_a_summer.shape
            outdims = (indims[0],indims[1],len(latvec),len(lonvec))

            # mask

            sstdata = pulldata("tos",ssp)

            oceanmask = sstdata.isel(time=0,variant=0)
            oceanmask = xr.where(np.isnan(oceanmask),1,0)

            all_a_summer_masked = all_a_summer.where(oceanmask==1)
            all_b_summer_masked = all_b_summer.where(oceanmask==1)

            summer_mean = np.empty(outdims)

            for ilat,lat in enumerate(latvec):
                print("working on latitude="+str(lat))
                for ilon, lon in enumerate(lonvec):

                    # nanmean but only for boxes with >25% land coverage
                    maskmean = np.mean(oceanmask.sel(lat=slice(lat,lat+outres),lon=slice(lon,lon+outres)))

                    if maskmean>0.25:

                        if lat<0:
                            summersel = all_a_summer_masked
                        else:
                            summersel = all_b_summer_masked 
                        
                        # slice
                        summer_slice = summersel.sel(lat=slice(lat,lat+outres),lon=slice(lon,lon+outres))

                        # weight
                        weights = np.cos(np.deg2rad(summer_slice.lat))

                        summer_mean[:,:,ilat,ilon] = np.asarray(summer_slice.weighted(weights).mean(dim=("lat","lon"),skipna=True))

                    else:
                        summer_mean[:,:,ilat,ilon] = np.nan

            self.lat = latvec
            self.lon = lonvec

            time2 = time.time()

            print(f"{time2-time1:4f} seconds for " + ssp)
        
            with open(fileout,"wb") as f:
                pickle.dump([summer_mean,latvec,lonvec],f)
        
        else:
            print("output data done")
            with open(fileout,"rb") as f:
                allall = pickle.load(f)

                summer_mean = allall[0]
                self.lat = allall[1]
                self.lon = allall[2]
            
        self.summerdata = summer_mean

class GriddedMPIPrecip:
    def __init__(self,params):

        outres = params["outres"]
        ssp = params["ssp"] 
        timerange = params["timerange"]
        filefront = params["filefront"]
        var = "pr"
        preciproll = params["preciproll"]

        # fileout = "data/"+ var + "MPI_regridded_summertime_"+str(outres)+"x"+str(outres)+".pkl"
        fileout = "data/" + filefront + ssp + "_" + var + "_roll"+str(preciproll)+"_" +str(timerange[0])+"-"+str(timerange[1])+".pkl"
        filecheck = glob.glob(fileout)
        
        if len(filecheck)==0:

            print('starting regrid')
            time1 = time.time()

            alldata_in = pulldata(var,ssp)

            allrolling = alldata_in.rolling(time=preciproll).mean() # rolling mean over the "preciproll" period

            latvec = np.arange(-90,90,10)
            lonvec = np.arange(0,360,10)

            all_rolling = allrolling.sel(time=slice(str(timerange[0]),str(timerange[1])))

            indims = all_rolling.shape
            outdims = (indims[0],indims[1],len(latvec),len(lonvec))

            # mask

            sstdata = pulldata("tos","ssp245")

            oceanmask = sstdata.isel(time=0,variant=0)
            oceanmask = xr.where(np.isnan(oceanmask),1,0)

            all_rolling_masked = all_rolling.where(oceanmask==1)
            print('rolling')

            all_rolling_mean = np.empty(outdims)

            for ilat,lat in enumerate(latvec):
                print("working on latitude="+str(lat))
                for ilon, lon in enumerate(lonvec):

                    # nanmean but only for boxes with >25% land coverage
                    maskmean = np.mean(oceanmask.sel(lat=slice(lat,lat+outres),lon=slice(lon,lon+outres)))

                    if maskmean>0.5:
                        
                        all_rolling_slice = all_rolling_masked.sel(lat=slice(lat,lat+outres),lon=slice(lon,lon+outres))
                        # weight
                        weights = np.cos(np.deg2rad(all_rolling_slice.lat))

                        all_rolling_mean[:,:,ilat,ilon] = np.asarray(all_rolling_slice.weighted(weights).mean(dim=("lat","lon"),skipna=True))

                    else:
                        all_rolling_mean[:,:,ilat,ilon] = np.nan

            self.lat = latvec
            self.lon = lonvec

            time2 = time.time()

            print(f"{time2-time1:4f} seconds for " + ssp)
        
            with open(fileout,"wb") as f:
                pickle.dump([all_rolling_mean,latvec,lonvec],f)
        
        else:
            print("output data done")
            with open(fileout,"rb") as f:
                allall = pickle.load(f)

                all_rolling_mean = allall[0]
                self.lat = allall[1]
                self.lon = allall[2]
            
        self.all_rolling_mean = all_rolling_mean


class GriddedMPIAnnualMean:
    def __init__(self,params):
        
        inres = params["inres"]
        ssp = params["ssp"]
        timerange = params["timerange"]
        filefront = params["filefront"]
        var = params["inputvar"]

        fileout = "data/" + filefront +"annualmean_" + ssp + "_" + var + "_"+str(timerange[0])+"-"+str(timerange[1])+".pkl"
        filecheck = glob.glob(fileout)

        if len(filecheck)==0:

            time1 = time.time()

            alldata_in = pulldata(var,ssp)    
            alldata_regrid = regrid(alldata_in,inres)

            alldata_annualmean = alldata_regrid.groupby("time.year").mean()
            annual_mean = np.asarray(alldata_annualmean.sel(year=slice(timerange[0],timerange[1])))

            lat = np.asarray(alldata_annualmean.lat)
            lon = np.asarray(alldata_annualmean.lon)

            time2 = time.time()

            print(f"{time2-time1:4f} seconds for " + ssp)
        
            with open(fileout,"wb") as f:
                print('input data done')
                pickle.dump([annual_mean,lat,lon],f)
        
        else:
            with open(fileout,"rb") as f:
                allall=pickle.load(f)

            annual_mean = allall[0]
            lat = allall[1]
            lon = allall[2]    
        
        self.lat = lat
        self.lon = lon
        self.annual_mean = annual_mean

class MPIGlobalMeanTemperature:
    def __init__(self,params):

        ssp = params["ssp"]
        timerange = params["timerange"]
        filefront = params["filefront"]

        fileout = "data/" + filefront +"annualmeanGMT_" + ssp + "_"+str(timerange[0])+"-"+str(timerange[1])+".pkl"
        filecheck = glob.glob(fileout)

        if len(filecheck)==0:

            time1 = time.time()

            alldata_in = pulldata('tas',ssp)    
            alldata_annualmean = alldata_in.groupby("time.year").mean()

            alldata_weights = np.cos(np.deg2rad(alldata_annualmean.lat))
            alldata_weightedmean = alldata_annualmean.weighted(alldata_weights).mean(dim=("lat","lon"))
            alldata_weightedmean = np.asarray(alldata_weightedmean.sel(year=slice(timerange[0],timerange[1])))

            time2 = time.time()

            print(f"{time2-time1:4f} seconds for " + ssp)

            with open(fileout,'wb') as f:
                pickle.dump(alldata_weightedmean,f)       
        else:
            print('gmt done')
            with open(fileout,'rb') as f:
                alldata_weightedmean = pickle.load(f)

        self.gmt = alldata_weightedmean

class GriddedERA5Summer:
    def __init__(self,params):

        outres = params["outres"]
        timerange = params["timerange"]
        filefront = params["filefront"]
        var = params["outputvar"]
        var = eravardict[var]

        # fileout = "data/"+ var + "MPI_regridded_summertime_"+str(outres)+"x"+str(outres)+".pkl"
        fileout = "data/ERA5_summertime_" + var + "_1940-2025.pkl"
        filecheck = glob.glob(fileout)

        if len(filecheck)==0:

            time1 = time.time()
            alldata_in = pulldata_obs(var)
            alldata_in = alldata_in.transpose("valid_time","latitude","longitude")

            alldata_in = alldata_in.sortby('latitude')

            allseasonal = alldata_in.rolling(valid_time=3).mean() # seasonal

            allseasonal = allseasonal.groupby("valid_time.month")

            all_a_summer = allseasonal[2]
            all_b_summer = allseasonal[8]

            latvec = np.arange(-90,90,10)
            lonvec = np.arange(0,360,10)

            all_a_summer = all_a_summer.sel(valid_time=slice(str(timerange[0]),str(2025)))
            all_b_summer = all_b_summer.sel(valid_time=slice(str(timerange[0]),str(2025)))

            indims = all_a_summer.shape
            outdims = (indims[0],len(latvec),len(lonvec)) # time x lat x lon

            # mask

            sstdata = pulldata_obs("sst")
            sstdata = sstdata.transpose("valid_time","latitude","longitude")
            sstdata = sstdata.sortby('latitude')

            oceanmask = sstdata.isel(valid_time=0)
            oceanmask = xr.where(np.isnan(oceanmask),1,0)

            all_a_summer_masked = all_a_summer.where(oceanmask==1)
            all_b_summer_masked = all_b_summer.where(oceanmask==1)
            summer_mean = np.empty(outdims)

            for ilat,lat in enumerate(latvec):
                print("working on latitude="+str(lat))
                for ilon, lon in enumerate(lonvec):

                    # nanmean but only for boxes with >25% land coverage
                    maskmean = np.mean(oceanmask.sel(latitude=slice(lat,lat+outres),longitude=slice(lon,lon+outres)))

                    if maskmean>0.25:

                        if lat<0:
                            summersel = all_a_summer_masked
                        else:
                            summersel = all_b_summer_masked 
                        
                        # slice
                        summer_slice = summersel.sel(latitude=slice(lat,lat+outres),longitude=slice(lon,lon+outres))

                        # weight
                        weights = np.cos(np.deg2rad(summer_slice.latitude))

                        summer_mean[:,ilat,ilon] = np.asarray(summer_slice.weighted(weights).mean(dim=("latitude","longitude"),skipna=True))

                    else:
                        summer_mean[:,ilat,ilon] = np.nan

            self.lat = latvec
            self.lon = lonvec

            time2 = time.time()

            print(f"{time2-time1:4f} seconds for ERA5")
        
            with open(fileout,"wb") as f:
                pickle.dump([summer_mean,latvec,lonvec],f)
        
        else:
            print("output data done")
            with open(fileout,"rb") as f:
                allall = pickle.load(f)

                summer_mean = allall[0]
                self.lat = allall[1]
                self.lon = allall[2]
            
        self.summerdata = summer_mean

class GriddedERA5AnnualMean:
    def __init__(self,params):
        
        inres = params["inres"]
        timerange = params["timerange"]
        filefront = params["filefront"]
        var = params["inputvar"]
        var = eravardict[var]

        fileout = "data/ERA5_annualmean_" + var + "_1940-2025.pkl"
        filecheck = glob.glob(fileout)

        if len(filecheck)==0:

            time1 = time.time()

            alldata_in = pulldata_obs(var)    
            alldata_regrid = regrid(alldata_in,inres)

            alldata_annualmean = alldata_regrid.groupby("valid_time.year").mean()
            annual_mean = np.asarray(alldata_annualmean.sel(year=slice(timerange[0],2025)))

            lat = np.asarray(alldata_annualmean.latitude)
            lon = np.asarray(alldata_annualmean.longitude)

            time2 = time.time()

            print(f"{time2-time1:4f} seconds for ERA5")
        
            with open(fileout,"wb") as f:
                print('input data done')
                pickle.dump([annual_mean,lat,lon],f)
        
        else:
            with open(fileout,"rb") as f:
                allall=pickle.load(f)

            annual_mean = allall[0]
            lat = allall[1]
            lon = allall[2]    
        
        self.lat = lat
        self.lon = lon
        self.annual_mean = annual_mean

class ERA5GlobalMeanTemperature:
    def __init__(self,params):

        timerange = params["timerange"]
        filefront = params["filefront"]

        fileout = "data/ERA5_annualmeanGMT_1940-2025.pkl"
        filecheck = glob.glob(fileout)

        if len(filecheck)==0:

            time1 = time.time()

            alldata_in = pulldata_obs('t2m')    
            alldata_annualmean = alldata_in.groupby("valid_time.year").mean()

            alldata_weights = np.cos(np.deg2rad(alldata_annualmean.latitude))
            alldata_weightedmean = alldata_annualmean.weighted(alldata_weights).mean(dim=("latitude","longitude"))
            alldata_weightedmean = np.asarray(alldata_weightedmean.sel(year=slice(timerange[0],2025)))

            time2 = time.time()

            print(f"{time2-time1:4f} seconds for ERA5")

            with open(fileout,'wb') as f:
                pickle.dump(alldata_weightedmean,f)       
        else:
            print('gmt done')
            with open(fileout,'rb') as f:
                alldata_weightedmean = pickle.load(f)

        self.gmt = alldata_weightedmean


def regrid(da,res):
    outgrid = xr.Dataset(
    {
        "lat": (["lat"], np.arange(-90+res/2,90+res/2, res), {"units": "degrees_north"}),
        "lon": (["lon"], np.arange(0+res/2, 360+res/2, res), {"units": "degrees_east"}),
    }
    )
    regridder = xe.Regridder(da, outgrid, "bilinear", periodic=True, ignore_degenerate=True)
    da_regrid = regridder(da,keep_attrs=True) 
    return da_regrid        

def pulldata(var,ssp):

    filename = glob.glob("data/"+ var + "*" + ssp +"*.nc")[0]

    ds = xr.open_dataset(filename)
    da = ds[var]

    return da

def pulldata_obs(var):

    filename = "data/ERA5_1940-2025.nc"

    ds = xr.open_dataset(filename)
    da = ds[var]

    return da
