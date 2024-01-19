#!/usr/bin/env python3
import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import xarray as xr
from pandas import date_range

import sys
sys.path.append('.')
from lfca_xarray import *

cutoff = 120 # low-pass cutoff in months
truncation = 30 # number of EOFs retained

# Load and pre-process data
filename = 'ERSST_1900_2016.mat'
mat = io.loadmat(filename)

lat_axis = mat['LAT_AXIS'][0]
lon_axis = mat['LON_AXIS'][:,0]
sst_mat = mat['SST']

time = date_range('1900-01-31','2016-12-31',freq='M')
sst=xr.DataArray(
    data=sst_mat,
    dims=['lon','lat','time'],
    coords={'lon': lon_axis,
            'lat': lat_axis,
            'time': time
           }
)
nlon, nlat, ntime = sst.shape

# Compute anomalies (remove monthly means)
sst_anomalies=(sst.groupby('time.month')-sst.groupby('time.month').mean()).drop('month')

# Weights for EOFs (proportional to square root of grid cell area)
weights = np.sqrt((np.cos(sst.lat/180*np.pi)).expand_dims({'lon':sst.lon}))
weights = weights.where(~sst.isel(time=0,drop=True).isnull(),0)

s = sst_anomalies.values.shape
y, x = np.meshgrid(lat_axis,lon_axis)
area = np.cos(y*np.pi/180.)
area[np.where(np.isnan(np.mean(sst_anomalies,-1)))] = 0

# Domain definition
domain = np.ones((nlon,nlat))
domain[np.where(x<100)] = 0
domain[np.where((x<103) & (y<5))] = 0
domain[np.where((x<105) & (y<2))] = 0
domain[np.where((x<111) & (y<-6))] = 0
domain[np.where((x<114) & (y<-7))] = 0
domain[np.where((x<127) & (y<-8))] = 0
domain[np.where((x<147) & (y<-18))] = 0
domain[np.where(y>70)] = 0
domain[np.where((y>65) & ((x<175) | (x>200)))] = 0
domain[np.where(y<-45)] = 0
domain[np.where((x>260) & (y>17))] = 0
domain[np.where((x>270) & (y<=17) & (y>14))] = 0
domain[np.where((x>276) & (y<=14) & (y>9))] = 0
domain[np.where((x>290) & (y<=9))] = 0

domain_xr=xr.DataArray(
    data=domain,
    dims=['lon','lat'],
    coords={'lon': lon_axis,
            'lat': lat_axis
           }
)

print('Starting LFCA...')
lfca_sst = Lfca(sst_anomalies, weights, domain_xr)
lfcs, lfps = lfca_sst.lfca(cutoff, truncation)

