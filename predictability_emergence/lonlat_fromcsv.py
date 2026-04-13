import pandas as pd
import numpy as np

def lons_lats(csvfile):
    """
    Get the array of longitudes and latitudes from the CSV file
    """
    df = pd.read_csv(csvfile)
    lonvec = np.sort(df.LON.unique())
    latvec = np.sort(df.LAT.unique())
    return lonvec, latvec
