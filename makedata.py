#%%
import DataMakar
import sys

# some exp params
# inputlength = 10
# outputavgtime = 3
outres = 10
timerange = [1900,2100]
filefront = "MPI_"
inres = 4
inputvar = 'tos'
outputvar = 'tas'
preciproll = 12
# user defined ssp
ssp = str(sys.argv[1])

params = {
    # "inputlength": inputlength,
    # "outputavgtime": outputavgtime,
    "outres": outres,
    "ssp": ssp,
    "timerange": timerange,
    "filefront": filefront,
    "inres": inres,
    "inputvar":inputvar,
    "outputvar":outputvar,
    "preciproll":preciproll,
}

# AllInputData = DataMakar.GriddedMPIAnnualMean(params)
# AllOutputData = DataMakar.GriddedMPISummer(params)
# AllOutputpr = DataMakar.GriddedMPIPrecip(params)
# AllGMTData = DataMakar.MPIGlobalMeanTemperature(params)

AllInputERA = DataMakar.GriddedERA5AnnualMean(params)
AllOutputERA = DataMakar.GriddedERA5Summer(params)
AllGMTERA = DataMakar.ERA5GlobalMeanTemperature(params)

# %%
