
import DataMaker
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
}

AllInputData = DataMaker.GriddedMPIAnnualMean(params)
AllOutputData = DataMaker.GriddedMPISummer(params)
AllGMTData = DataMaker.MPIGlobalMeanTemperature(params)
