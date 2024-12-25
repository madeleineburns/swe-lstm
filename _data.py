# _data.py contains functions that support data management, preparation of training data for the LSTM model, and preparation of data as well as running the ParFlow-CLM model.


## PRELIMINARIES ##
# general 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import io
import xarray as xr
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy import stats
import random as rn
from captum.attr import IntegratedGradients
import pickle
from glob import glob

# for data
import subsettools as st
import hf_hydrodata as hf
from pandas.tseries.offsets import DateOffset

# for parflow
from parflow import Run
import shutil
import parflow as pf
from parflow.tools.fs import mkdir, cp, chdir, get_absolute_path, rm, exists
import parflow.tools.hydrology as hydro
import netCDF4

from contextlib import redirect_stdout
trap = io.StringIO()

from _lstm import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

## WATER YEAR START ##
# given datetime date, return start of next water year
def get_wy_start(date, site_id):
    try:
        with redirect_stdout(trap):
            site_df_swe = hf.get_point_data(dataset='snotel', variable='swe', temporal_resolution='daily', aggregation='sod', site_ids=[site_id])
            site_df_precip = hf.get_point_data(dataset='snotel', variable='precipitation', temporal_resolution='daily', aggregation='sum', site_ids=[site_id])
            site_df_temp = hf.get_point_data(dataset='snotel', variable='air_temp', temporal_resolution='daily', aggregation='mean', site_ids=[site_id])
    except:
        return np.nan

    date = max([pd.to_datetime(site_df_swe['date'][0]), pd.to_datetime(site_df_temp['date'][0]), pd.to_datetime(site_df_precip['date'][0])])
    
    if (date.month == 10) and (date.day == 1):
        data_start = date.strftime('%Y-%m-%d')
    elif date.month < 10:
        data_start = str(date.year)+'-10-01'
    else:
        data_start = str(date.year+1)+'-10-01'
    return data_start

## WATER YEAR END ##
# given datetime date, return end of next water year
# compatible with CW3E (ends on WY 2022)
def get_wy_end(date, site_id):
    try:
        with redirect_stdout(trap):
            site_df_swe = hf.get_point_data(dataset='snotel', variable='swe', temporal_resolution='daily', aggregation='sod', site_ids=[site_id])
            site_df_precip = hf.get_point_data(dataset='snotel', variable='precipitation', temporal_resolution='daily', aggregation='sum', site_ids=[site_id])
            site_df_temp = hf.get_point_data(dataset='snotel', variable='air_temp', temporal_resolution='daily', aggregation='mean', site_ids=[site_id])
    except:
        return np.nan

    date = min([pd.to_datetime(site_df_swe['date'][len(site_df_swe)-1]), pd.to_datetime(site_df_temp['date'][len(site_df_temp)-1]), 
            pd.to_datetime(site_df_precip['date'][len(site_df_precip)-1])])
    
    if (date.month == 9) and (date.day == 30):
        data_end = date.strftime('%Y-%m-%d')
    elif date.month < 10:
        data_end = str(date.year-1)+'-09-30'
    else:
        data_end = str(date.year)+'-09-30'
    if(date.year == 2023) or (date.year == 2024):
        data_end = '2022-09-30'
    return data_end


## GET ALL SNOTEL SITES ##
def get_sites_full(num_sites, num_years):
    snotel = hf.get_site_variables(variable="swe")
    snotel = snotel.reset_index(drop=True).drop(columns=['variable_name','units','site_query_url','date_metadata_last_updated','tz_cd','doi'])

    # read in testing data and remove it from dataset
    data_test = pd.read_csv('national_test_sites.txt', sep=' ',header=None)
    data_test.columns = ['site_id', 'site_name', 'site_type', 'agency', 'state','first_date_data_available', 'last_date_data_available', 'record_count',
                         'latitude', 'longitude', 'bins', 'first_wy_date', 'last_wy_date']
    data_test = data_test.drop(columns=['bins', 'first_wy_date', 'last_wy_date'])
    snotel = snotel[~snotel['site_id'].isin(data_test['site_id'])].reset_index(drop=True)
    
    snotel['first_wy_date'] = snotel.apply(lambda x: get_wy_start(pd.to_datetime(x.first_date_data_available), x.site_id), axis=1)
    snotel['last_wy_date'] = snotel.apply(lambda x: get_wy_end(pd.to_datetime(x.last_date_data_available), x.site_id), axis=1)
    snotel = snotel.dropna(axis='index', subset=['first_wy_date','last_wy_date'], how='any').reset_index(drop=True)

    snotel['num years'] = np.array(list(int(x.split('-')[0]) for x in snotel['last_wy_date'])) - np.array(list(int(x.split('-')[0]) for x in snotel['first_wy_date']))
    snotel = snotel[snotel['num years'] >= num_years].reset_index(drop=True)

    if(num_sites > len(snotel)):
        print('requested sites:', num_sites, 'available sites: ', len(snotel))

    return snotel

## GET RANDOM SELECTION OF SNOTEL SITES ##
# for given state and number of sites
def get_sites_random(num_sites, num_years):
    snotel_full = hf.get_site_variables(variable="swe")
    snotel_full = snotel_full.reset_index(drop=True).drop(columns=['variable_name','units','site_query_url','date_metadata_last_updated','tz_cd','doi'])

    # read in testing data and remove it from dataset
    data_test = pd.read_csv('national_test_sites.txt', sep=' ',header=None)
    data_test.columns = ['site_id', 'site_name', 'site_type', 'agency', 'state','first_date_data_available', 'last_date_data_available', 'record_count',
                         'latitude', 'longitude', 'bins', 'first_wy_date', 'last_wy_date']
    data_test = data_test.drop(columns=['bins', 'first_wy_date', 'last_wy_date'])
    snotel_full = snotel_full[~snotel_full['site_id'].isin(data_test['site_id'])].reset_index(drop=True)

    snotel_full['first_wy_date'] = snotel_full.apply(lambda x: get_wy_start(pd.to_datetime(x.first_date_data_available), x.site_id), axis=1)
    snotel_full['last_wy_date'] = snotel_full.apply(lambda x: get_wy_end(pd.to_datetime(x.last_date_data_available), x.site_id), axis=1)
    snotel_full = snotel_full.dropna(axis='index', subset=['first_wy_date','last_wy_date'], how='any').reset_index(drop=True)
    snotel_full['num years'] = np.array(list(int(x.split('-')[0]) for x in snotel_full['last_wy_date'])) - np.array(list(int(x.split('-')[0]) for x in snotel_full['first_wy_date']))
    snotel_full = snotel_full[snotel_full['num years'] >= num_years].reset_index(drop=True)

    if(num_sites > len(snotel_full)):
        print('requested sites:', num_sites, 'available sites: ', len(snotel_full))
        return snotel_full

    # select a random selection of num_sites from snotel data
    rn.seed(8)
    sample = rn.sample(list(snotel_full.index), num_sites)
    snotel = snotel_full.iloc[sample].reset_index(drop=True)
    
    #snotel = snotel.dropna(axis='index', how='any').reset_index(drop=True)

    return snotel

## GET SELECTION OF SNOTEL SITES BASED ON LONGITUDE ONLY ##
# for given number of sites (divide by number of longitude bin, may not necessarily be accurate)
def get_sites_longitude(num_sites, num_years):  
    snotel_full = hf.get_site_variables(variable="swe")
    snotel_full = snotel_full.reset_index(drop=True).drop(columns=['variable_name','units','site_query_url','date_metadata_last_updated','tz_cd','doi'])

    #read in testing data and remove it from dataset
    data_test = pd.read_csv('national_test_sites.txt', sep=' ',header=None)
    data_test.columns = ['site_id', 'site_name', 'site_type', 'agency', 'state','first_date_data_available', 'last_date_data_available', 'record_count',
                         'latitude', 'longitude', 'bins', 'first_wy_date', 'last_wy_date']
    data_test = data_test.drop(columns=['bins', 'first_wy_date', 'last_wy_date'])
    snotel_full = snotel_full[~snotel_full['site_id'].isin(data_test['site_id'])].reset_index(drop=True)

    snotel_full['first_wy_date'] = snotel_full.apply(lambda x: get_wy_start(pd.to_datetime(x.first_date_data_available), x.site_id), axis=1)
    snotel_full['last_wy_date'] = snotel_full.apply(lambda x: get_wy_end(pd.to_datetime(x.last_date_data_available), x.site_id), axis=1)
    snotel_full = snotel_full.dropna(axis='index', subset=['first_wy_date','last_wy_date'], how='any').reset_index(drop=True)
    snotel_full['num years'] = np.array(list(int(x.split('-')[0]) for x in snotel_full['last_wy_date'])) - np.array(list(int(x.split('-')[0]) for x in snotel_full['first_wy_date']))
    snotel_full = snotel_full[snotel_full['num years'] >= num_years].reset_index(drop=True)

    if(num_sites > len(snotel_full)):
        print('requested sites:', num_sites, 'available sites: ', len(snotel_full))
    
    rn.seed(8)
    
    # FIRST: bin snotel sites into 3 based on longitude
    bins = [-125, -118, -111, -100]
    bin_labels=['maritime','intermountain','continental'] 
    snotel_full['bins'] = pd.cut(snotel_full['longitude'], bins=bins, labels=bin_labels, include_lowest=True)
    
    l_bins = []
    for b in bin_labels:
        size = int(np.ceil(num_sites/3))
        while size > 0:
            try:
                l_bins.extend(rn.sample(snotel_full.loc[snotel_full['bins'] == b].index.to_list(), size))
                break
            except:
                # if not enough sites exist, try to get mininum sample size
                size = size-1
                print('not enough sites in longitude ' + str(b) + '. try ' + str(size) +' sites.')
    snotel = snotel_full.iloc[l_bins].reset_index(drop=True)
    # snotel.loc[:,'first_wy_date'] = snotel.apply(lambda x: get_wy_start(pd.to_datetime(x.first_date_data_available), x.site_id), axis=1)
    # snotel.loc[:,'last_wy_date'] = snotel.apply(lambda x: get_wy_end(pd.to_datetime(x.last_date_data_available), x.site_id), axis=1)
    # snotel = snotel.dropna(axis='index', how='any').reset_index(drop=True)

    return snotel


## GET SELECTION OF SNOTEL SITES BASED ON LONGITUDE AND PERCENTAGE OF MARITIME SITES ##
# if percent_maritime <= 1, uses selection by percent; if not, acts the same as get_sites_longitude
# defined percent of sites are maritime; remainder of sites are split between intermountain and continental 
def get_sites_longitude_bypercent(num_sites, num_years):
    percent_maritime = 0.3   # percent as decimal
    
    snotel_full = hf.get_site_variables(variable="swe")
    snotel_full = snotel_full.reset_index(drop=True).drop(columns=['variable_name','units','site_query_url','date_metadata_last_updated','tz_cd','doi'])
    
    #read in testing data and remove it from dataset
    data_test = pd.read_csv('national_test_sites.txt', sep=' ',header=None)
    data_test.columns = ['site_id', 'site_name', 'site_type', 'agency', 'state','first_date_data_available', 'last_date_data_available', 'record_count',
                 'latitude', 'longitude', 'bins', 'first_wy_date', 'last_wy_date']
    data_test = data_test.drop(columns=['bins', 'first_wy_date', 'last_wy_date'])
    snotel_full = snotel_full[~snotel_full['site_id'].isin(data_test['site_id'])].reset_index(drop=True)
    
    snotel_full['first_wy_date'] = snotel_full.apply(lambda x: get_wy_start(pd.to_datetime(x.first_date_data_available), x.site_id), axis=1)
    snotel_full['last_wy_date'] = snotel_full.apply(lambda x: get_wy_end(pd.to_datetime(x.last_date_data_available), x.site_id), axis=1)
    snotel_full = snotel_full.dropna(axis='index', subset=['first_wy_date','last_wy_date'], how='any').reset_index(drop=True)
    snotel_full['num years'] = np.array(list(int(x.split('-')[0]) for x in snotel_full['last_wy_date'])) - np.array(list(int(x.split('-')[0]) for x in snotel_full['first_wy_date']))
    snotel_full = snotel_full[snotel_full['num years'] >= num_years].reset_index(drop=True)
    
    if(num_sites > len(snotel_full)):
        print('requested sites:', num_sites, 'available sites: ', len(snotel_full))
    
    rn.seed(8)
    
    # FIRST: bin snotel sites into 3 based on longitude
    bins = [-125, -118, -111, -100]
    bin_labels=['maritime','intermountain','continental'] 
    snotel_full['bins'] = pd.cut(snotel_full['longitude'], bins=bins, labels=bin_labels, include_lowest=True)

    l_bins = []
    for b in bin_labels:
        if(percent_maritime <= 1):
            if b == 'maritime':
                size = int(percent_maritime * num_sites)
            else:
                size = int((1 - percent_maritime)*num_sites/2)
        else:
            size = int(np.ceil(num_sites/3))
        while size > 0:
            try:
                l_bins.extend(rn.sample(snotel_full.loc[snotel_full['bins'] == b].index.to_list(), size))
                break
            except:
                # if not enough sites exist, try to get mininum sample size
                size = size-1
                print('not enough sites in longitude ' + str(b) + '. try ' + str(size) +' sites.')
    snotel = snotel_full.iloc[l_bins].reset_index(drop=True)
    return snotel

## GET SELECTION OF SNOTEL SITES BASED ON LATITUDE and LONGITUDE ##
# for given number of sites (divide by number of latitude bin, may not necessarily be accurate)
# must be a multiple of 20
def get_sites_latitude(num_sites, num_years): 
    if((num_sites % 20) != 0):
        print("number of sites must be a multiple of 20")
        
    snotel_full = hf.get_site_variables(variable="swe")
    snotel_full = snotel_full.reset_index(drop=True).drop(columns=['variable_name','units','site_query_url','date_metadata_last_updated','tz_cd','doi'])

    # read in testing data and remove it from dataset
    data_test = pd.read_csv('national_test_sites.txt', sep=' ',header=None)
    data_test.columns = ['site_id', 'site_name', 'site_type', 'agency', 'state','first_date_data_available', 'last_date_data_available', 'record_count',
                         'latitude', 'longitude', 'bins', 'first_wy_date', 'last_wy_date']
    data_test = data_test.drop(columns=['bins', 'first_wy_date', 'last_wy_date'])
    snotel_full = snotel_full[~snotel_full['site_id'].isin(data_test['site_id'])].reset_index(drop=True)

    snotel_full['first_wy_date'] = snotel_full.apply(lambda x: get_wy_start(pd.to_datetime(x.first_date_data_available), x.site_id), axis=1)
    snotel_full['last_wy_date'] = snotel_full.apply(lambda x: get_wy_end(pd.to_datetime(x.last_date_data_available), x.site_id), axis=1)
    snotel_full = snotel_full.dropna(axis='index', subset=['first_wy_date','last_wy_date'], how='any').reset_index(drop=True)
    snotel_full['num years'] = np.array(list(int(x.split('-')[0]) for x in snotel_full['last_wy_date'])) - np.array(list(int(x.split('-')[0]) for x in snotel_full['first_wy_date']))
    snotel_full = snotel_full[snotel_full['num years'] >= num_years].reset_index(drop=True)

    if(num_sites > len(snotel_full)):
        print('requested sites:', num_sites, 'available sites: ', len(snotel_full))
        return snotel
    
    rn.seed(8)
    
    # FIRST: bin snotel sites into 3 based on longitude
    bins = [-125, -118, -111, -100]
    bin_labels=['maritime','intermountain','continental'] 
    snotel_full['bins'] = pd.cut(snotel_full['longitude'], bins=bins, labels=bin_labels, include_lowest=True)
    
    l_bins = []
    for b in bin_labels:
        size = int(np.ceil(num_sites/3))
        while size > 0:
            try:
                l_bins.extend(rn.sample(snotel_full.loc[snotel_full['bins'] == b].index.to_list(), size))
                break
            except:
                # if not enough sites exist, try to get mininum sample size
                size = size-1
                print('not enough sites in longitude ' + str(b) + '. try ' + str(size) +' sites.')
    snotel_lon = snotel_full.iloc[l_bins].reset_index(drop=True)
    
    # SECOND: bin snotel sites based on latitude - 20 bins
    bins = np.arange(0,20)
    snotel_lon['bins'] = pd.qcut(snotel_lon['latitude'], q=20, labels=bins[:len(bins)])
    
    l_bins = []
    for b in bins[0:len(bins)-1]:
        size = int(np.ceil(num_sites/20))
        while size > 0:
            try:
                l_bins.extend(rn.sample(snotel_lon.loc[snotel_lon['bins'] == b].index.to_list(), size)) 
                break
            except:
                # if not enough sites exist, try to get mininum sample size
                size = size-1
                print('not enough sites in latitude ' + str(b) + '. try ' + str(size) +' sites.')
    
    snotel = snotel_lon.iloc[l_bins].reset_index(drop=True)
    # snotel.loc[:,'first_wy_date'] = snotel.apply(lambda x: get_wy_start(pd.to_datetime(x.first_date_data_available), x.site_id), axis=1)
    # snotel.loc[:,'last_wy_date'] = snotel.apply(lambda x: get_wy_end(pd.to_datetime(x.last_date_data_available), x.site_id), axis=1)
    # snotel = snotel.dropna(axis='index', how='any').reset_index(drop=True).drop(columns=[level_0'])

    return snotel

## GET SINGLE COLUMN DATA ##
# returns SNOTEL and CW3E data for point location (snotel site with site_id)
# NOT formatted for input into LSTM: just a non-normalized 2D array
# assume start_date and end_date are water year adjusted
def get_sc_data(site_id, start_date, end_date):
    # GET SNOTEL DATA 
    site_df_swe = hf.get_point_data(dataset='snotel', variable='swe', temporal_resolution='daily', aggregation='sod', 
                                             date_start=start_date, date_end=end_date, site_ids=[site_id])
    site_df_precip = hf.get_point_data(dataset='snotel', variable='precipitation', temporal_resolution='daily', aggregation='sum', 
                                   date_start=start_date, date_end=end_date, site_ids=[site_id])
    site_df_temp = hf.get_point_data(dataset='snotel', variable='air_temp', temporal_resolution='daily', aggregation='mean', 
                                 date_start=start_date, date_end=end_date, site_ids=[site_id])
    metadata_df = hf.get_point_metadata(dataset='snotel', variable='swe', temporal_resolution='daily', aggregation='sod', 
                                             date_start=start_date, date_end=end_date, site_ids=[site_id])
    print('loaded SNOTEL data')
        
    # PARAMETERS FOR CW3E DATA
    precip = site_df_precip.set_axis(['date','precip'], axis='columns')
    temp = site_df_temp.set_axis(['date','temp'], axis='columns')
    tot_swe = site_df_swe.set_axis(['date','swe'], axis='columns')

    # adjust end date for CW3E
    end_date_cw3e = str(pd.to_datetime(end_date) + DateOffset(days=1))
    wy = str(pd.to_datetime(end_date).year)
    
    lat = metadata_df['latitude'][0]               	
    lon = metadata_df['longitude'][0]              
    bounds = st.latlon_to_ij([[lat, lon],[lat, lon]],"conus2")
    
    # GET DAILY CW3E DATA
    # APCP = hf.get_gridded_data({"dataset":"CW3E", "period":"daily", "variable":"precipitation", "start_time":data_start, "end_time":data_end_cw3e,
    #                             "grid_bounds":bounds}) 
    # DLWR = hf.get_gridded_data({"dataset":"CW3E", "period":"daily", "variable":"downward_longwave", "start_time":data_start, "end_time":data_end_cw3e,
    #                             "grid_bounds":bounds})  
    # DSWR = hf.get_gridded_data({"dataset":"CW3E", "period":"daily", "variable":"downward_shortwave", "start_time":data_start, "end_time":data_end_cw3e,
    #                             "grid_bounds":bounds}) 
    # SPFH = hf.get_gridded_data({"dataset":"CW3E", "period":"daily", "variable":"specific_humidity", "start_time":data_start, "end_time":data_end_cw3e,
    #                             "grid_bounds":bounds}) 
    # Temp = hf.get_gridded_data({"dataset":"CW3E", "period":"daily", "variable":"air_temp", "start_time":data_start, "end_time":data_end_cw3e, 
    #                             "aggregation":"mean","grid_bounds":bounds}) 
    # UGRD = hf.get_gridded_data({"dataset":"CW3E", "period":"daily", "variable":"east_windspeed", "start_time":data_start, "end_time":data_end_cw3e,
    #                             "grid_bounds":bounds}) 
    # VGRD = hf.get_gridded_data({"dataset":"CW3E", "period":"daily", "variable":"north_windspeed", "start_time":data_start, "end_time":data_end_cw3e,
    #                             "grid_bounds":bounds}) 
    # Press = hf.get_gridded_data({"dataset":"CW3E", "period":"daily", "variable":"atmospheric_pressure", "start_time":data_start, "end_time":data_end_cw3e,
    #                              "grid_bounds":bounds}) 
    # met_data = pd.DataFrame({"DSWR":DSWR[:,0,0], "DLWR":DLWR[:,0,0], "precip":APCP[:,0,0], "temp":Temp[:,0,0], "wind (E)":UGRD[:,0,0], "wind (N)":VGRD[:,0,0],
    #                          "pressure":Press[:,0,0], "q":SPFH[:,0,0]})
    variables = ["precipitation", "downward_longwave", "downward_shortwave", "specific_humidity", "air_temp", "east_windspeed", "north_windspeed", 
                 "atmospheric_pressure"]

    if (not os.path.exists('Data/LSTM_training/'+site_id+'_'+wy+'.nc')):
        hf.get_gridded_files({"dataset": "CW3E", "temporal_resolution": "daily", "start_time": start_date, "end_time": end_date_cw3e, 
                              "grid_point": [bounds[0], bounds[1]]}, variables=variables, filename_template="Data/LSTM_training/"+site_id+"_{wy}.nc")
    try:
        with xr.open_dataset('Data/LSTM_training/'+site_id+'_'+wy+'.nc', engine='netcdf4', 
                         drop_variables=['y', 'x','latitude','longitude', 'Temp_min', 'Temp_max']) as ds:
            df = ds.to_dataframe()
    except:
        os.remove('Data/LSTM_training/'+site_id+'_'+wy+'.nc')
        with xr.open_dataset('Data/LSTM_training/'+site_id+'_'+wy+'.nc', engine='netcdf4', 
                         drop_variables=['y', 'x','latitude','longitude', 'Temp_min', 'Temp_max']) as ds:
            df = ds.to_dataframe()

    # if(df.size != 365):
    #     os.remove('Data/LSTM_training/'+site_id+'_'+wy+'.nc')
    #     hf.get_gridded_files({"dataset": "CW3E", "temporal_resolution": "daily", "start_time": start_date, "end_time": end_date_cw3e, 
    #                           "grid_point": [bounds[0], bounds[1]]}, variables=variables, filename_template="Data/LSTM_training/"+site_id+"_{wy}.nc")
    #     ds = xr.open_dataset('Data/LSTM_training/'+site_id+'_'+wy+'.nc', engine='netcdf4', drop_variables=['y', 'x','latitude','longitude', 'Temp_min', 'Temp_max'])
    #     df = ds.to_dataframe()
        
    
    met_data = pd.DataFrame({"DSWR":df['DSWR'].values, "DLWR":df['DLWR'].values, "precip":df['APCP'].values, "temp":df['Temp_mean'].values, 
                             "wind (E)":df['UGRD'].values, "wind (N)":df['VGRD'].values,"pressure":df['Press_atmos'].values, "q":df['SPFH'].values})
    print('loaded CW3E data')

    ## GET TOPOGRAPHICAL DATA ##
    # land_cover = hf.get_gridded_data({"dataset": "conus2_domain", "variable": "veg_type_IGBP", "grid": "conus2", "start_time":data_start,"end_time":data_end,
    #                                   "grid_bounds":bounds})
    # slope_x = hf.get_gridded_data({"dataset": "conus2_domain", "variable": "slope_x", "grid": "conus2", "start_time":data_start,"end_time":data_end,
    #                                   "grid_bounds":bounds})
    # slope_y = hf.get_gridded_data({"dataset": "conus2_domain", "variable": "slope_y", "grid": "conus2", "start_time":data_start,"end_time":data_end,
    #                                   "grid_bounds":bounds})
    # top_data = pd.DataFrame({"land_cover":land_cover[0][0][0], "slope_x":slope_x[0][0], "slope_y":slope_y[0][0]}, index=[0])
    variables = ["veg_type_IGBP", "slope_x", "slope_y"]
    #if (not os.path.exists('Data/LSTM_training/'+site_id+'_'+wy+'_static.nc')):
        #os.remove('Data/LSTM_training/'+site_id+'_'+wy+'_static.nc')

    try:
        ds = xr.open_dataset('Data/LSTM_training/'+site_id+'_'+wy+'_static.nc', engine='netcdf4', drop_variables=['y', 'x','latitude','longitude'])
    except:
        if (os.path.exists('Data/LSTM_training/'+site_id+'_'+wy+'_static.nc')):
            os.remove('Data/LSTM_training/'+site_id+'_'+wy+'_static.nc')
        hf.get_gridded_files({"dataset": "conus2_domain", "grid":"conus2", "start_time": start_date, "end_time": end_date_cw3e, "grid_point": [bounds[0], 
                                                                                                                                           bounds[1]]}, 
                         variables=variables, filename_template="Data/LSTM_training/"+site_id+"_{wy}_static.nc")
        ds = xr.open_dataset('Data/LSTM_training/'+site_id+'_'+wy+'_static.nc', engine='netcdf4', drop_variables=['y', 'x','latitude','longitude'])
    df = ds.to_dataframe()
    
    top_data = pd.DataFrame({"land_cover":df['vegetation_type'].values, "slope_x":df['slope_x'].values, "slope_y":df['slope_y'].values}, index=[0])
    print('loaded topographical data')
    
    # PROCESS
    # precip: fill nan values with zero
    #precip_start = precip.loc[precip['date'] == start_date].index[0]
    #precip_end = precip.loc[precip['date'] == end_date].index[0]
    #precip = precip[precip_start:precip_end+1].reset_index(drop=True)
    precip['year'] = pd.DatetimeIndex(precip['date']).year
    precip = precip.set_index('date')
    precip['precip'] = precip['precip'].fillna(0)
    cols = precip.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    precip = precip[cols]
    
    # temp: fill nan values with linear interpolation
    #temp_start = temp.loc[temp['date'] == start_date].index[0]
    #temp_end = temp.loc[temp['date'] == end_date].index[0]
    #temp = temp[temp_start:temp_end+1].reset_index(drop=True)
    temp = temp.set_index('date')
    temp['temp'] = temp['temp'].interpolate(method='linear', limit_direction='both')
    
    # swe: fill nan values with linear interpolation
    #swe_start = tot_swe.loc[tot_swe['date'] == start_date].index[0]
    #swe_end = tot_swe.loc[tot_swe['date'] == end_date].index[0]
    #tot_swe = tot_swe[swe_start:swe_end+1].reset_index(drop=True)
    tot_swe['year'] = pd.DatetimeIndex(tot_swe['date']).year
    tot_swe = tot_swe.set_index('date')
    tot_swe['swe'] = tot_swe['swe'].interpolate(method='linear', limit_direction='both')
    
    # count years
    years = np.unique(tot_swe['year'])
    years = years[~np.isnan(years)]
    num_values = len(years)
    
    # combine all datasets and add SNOTEL metadata characteristitcs
    tot_non_swe = precip.join(temp)
    tot_non_swe['elevation'] = metadata_df['usda_elevation'][0]
    tot_non_swe['latitude'] = metadata_df['latitude'][0]
    tot_non_swe['longitude'] = metadata_df['longitude'][0]

    # add CW3E forcing characteristics
    tot_non_swe['DSWR'] = met_data['DSWR'].to_numpy().reshape(-1,1)
    tot_non_swe['DLWR'] = met_data['DLWR'].to_numpy().reshape(-1,1)
    tot_non_swe['wind (E)'] = met_data['wind (E)'].to_numpy().reshape(-1,1)
    tot_non_swe['wind (N)'] = met_data['wind (N)'].to_numpy().reshape(-1,1)
    tot_non_swe['pressure'] = met_data['pressure'].to_numpy().reshape(-1,1)
    tot_non_swe['q'] = met_data['q'].to_numpy().reshape(-1,1)

    # add topographic forcing characteristics
    tot_non_swe['land_cover'] = top_data['land_cover'][0]
    tot_non_swe['slope_x'] = top_data['slope_x'][0]
    tot_non_swe['slope_y'] = top_data['slope_y'][0]

    # last check for nan values
    tot_non_swe = tot_non_swe.fillna(0)
    
    # change years to water years - assume dates of swe/non swe are the same
    leap=[]
    for i in range(0,len(tot_swe)-1):
        d_non_swe = pd.to_datetime(tot_non_swe.index[i])
        d_swe = pd.to_datetime(tot_swe.index[i])
        if(d_non_swe.month >= 10):
            tot_non_swe.loc[tot_non_swe.index[i],'year'] += 1
        if(d_swe.month >= 10):
            tot_swe.loc[tot_swe.index[i],'year'] += 1
        if((d_non_swe.month==2) and (d_non_swe.day==29)):
            leap.append(tot_swe.index[i])
            
    # get rid of leap days
    tot_swe = tot_swe.drop(leap, axis=0)
    tot_non_swe = tot_non_swe.drop(leap, axis=0)

    # add site_id, reindex to remove date
    tot_non_swe['site_id'] = site_id
    tot_swe['site_id'] = site_id
    tot_non_swe = tot_non_swe.reset_index(drop=True)
    tot_swe = tot_swe.reset_index(drop=True)

    return tot_swe, tot_non_swe


## GET NORMALIZATION FUNCTIONS ##
# return list of normalization functions, SWE as the first one
def create_normalization(tot_swe, tot_non_swe):
    l_normalize = []
    # for swe variables - define scaling object for swe so data can be inverted later
    scaler_swe = MaxAbsScaler().fit(tot_swe[['swe']])
    l_normalize.append(scaler_swe)

    # for non-swe variables
    # ignore 0 (the year) and last (the site_id)
    for i in range(1,len(tot_non_swe.columns)-1):
        variable = tot_non_swe.columns[i]
        scaler = MaxAbsScaler().fit(tot_non_swe[[variable]])
        l_normalize.append(scaler)

    return l_normalize

## GENERATE DATA FOR TRAINING/TESTING ##
# return 3D array of tensors, with data organized by year
def create_dataset(tot_swe, tot_non_swe, l_normalize):
    # NORMALIZE DATA
    # for non-swe variables
    for i in range(1,len(tot_non_swe.columns)-1):
        variable = tot_non_swe.columns[i]
        scaler = l_normalize[i]
        tot_non_swe[variable] = scaler.transform(tot_non_swe[[variable]])

    # for swe variables
    scaler_swe = l_normalize[0]
    tot_swe['swe'] = scaler_swe.transform(tot_swe[['swe']])
    
    # CREATE DATASETS
    # add in processing by site_id
    sites, ind = np.unique(tot_swe['site_id'], return_index=True)
    sites = sites[np.argsort(ind)]
    
    l_swe = []
    l_non_swe = []
    l_sites = []
    l_years = []
    for j in range(0, len(sites)):
        site_swe = tot_swe.loc[tot_swe['site_id'] == sites[j]]
        site_swe = site_swe.loc[:, site_swe.columns != 'site_id']
        site_non_swe = tot_non_swe.loc[tot_non_swe['site_id'] == sites[j]]
        site_non_swe = site_non_swe.loc[:, site_non_swe.columns != 'site_id']
        years, ind = np.unique(site_swe['year'], return_index=True)
        years = years[np.argsort(ind)]
        years = years[~np.isnan(years)]
        
        for i in range(0, len(years)):
            year_swe = site_swe.loc[site_swe['year'] == years[i]]
            temp_swe = year_swe.loc[:, year_swe.columns != 'year'].to_numpy()
            year_non_swe = site_non_swe.loc[site_non_swe['year'] == years[i]]
            temp_non_swe = year_non_swe.loc[:, year_non_swe.columns != 'year'].to_numpy()
            
            # get rid of years without 365 days of data
            # consider making a more nuanced way to filter data?
            if((temp_swe.size == (365*temp_swe.shape[1])) and (temp_non_swe.size == (365*temp_non_swe.shape[1]))):
                l_swe.append(temp_swe)
                l_non_swe.append(temp_non_swe)
                l_sites.append(sites[j])
                l_years.append(years[i])
        
    # create arrays with all years
    full_swe = np.stack(l_swe).astype(np.float32)
    full_non_swe = np.stack(l_non_swe).astype(np.float32)
    full_sites = np.stack(l_sites)
    full_years = np.stack(l_years).astype(np.int_)
    num_years = len(full_years)

    full_swe_tensors = torch.from_numpy(full_swe)
    full_non_swe_tensors = torch.from_numpy(full_non_swe)

    #return full_swe, full_non_swe, full_sites, full_years, num_years,
    return full_swe_tensors, full_non_swe_tensors, full_sites, full_years

## GET RANDOM DATA ##
# given a start and end point, randomly select sample_size years
def get_years_random(start_date, end_date, site_id, sample_size):
    # adjust for water year naming convention
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    start_year = start.year + 1
    end_year = end.year + 1
    
    years = np.arange(start_year, end_year).tolist()

    rn.seed(8)
    sample = rn.sample(years, sample_size)
    return sample

## GET DATA BINNED BY PRECIPITATION ##
# sample size is number of TOTAL samples
# split into below avg, avg, above avg swe
def get_years_precip(start_date, end_date, site_id, sample_size):
    sample_size = int(sample_size/3)
    # GET DATA
    swe = hf.get_point_data(dataset='snotel', variable='swe', temporal_resolution='daily', aggregation='sod', 
                                             date_start=start_date, date_end=end_date, site_ids=[site_id])
    swe.columns = ['date','swe']
    swe['year'] = pd.DatetimeIndex(swe['date']).year
    swe = swe.set_index('date')
    
    # change years to water years, get rid of leap years
    leap=[]
    for i in range(0,len(swe)-1):
        d_swe = pd.to_datetime(swe.index[i])
        if(d_swe.month >= 10):
            swe.loc[swe.index[i],'year'] += 1
        if((d_swe.month==2) and (d_swe.day==29)):
            leap.append(swe.index[i])
    swe = swe.drop(leap, axis=0)
    swe = swe.reset_index(drop=True)
    
    years = np.unique(swe['year'])

    # BIN SWE
    peak_values = np.column_stack((years, np.zeros(len(years))))
    for i in range(0,len(years)):
        yr = years[i]
        df_swe = swe.loc[swe['year'] == yr]
        peak_values[i,1] = max(df_swe['swe'])
    peak_values = pd.DataFrame(peak_values, columns = ['year','peak swe'])
    peak_values['bins'] = pd.qcut(peak_values['peak swe'], q=3, labels=['below avg', 'avg', 'above avg'])

    # RANDOMLY SELECT 
    rn.seed(6)
    abv_avg_i = rn.sample(peak_values.loc[peak_values['bins'] == 'above avg'].index.to_list(), sample_size)
    avg_i = rn.sample(peak_values.loc[peak_values['bins'] == 'avg'].index.to_list(), sample_size)
    bel_avg_i = rn.sample(peak_values.loc[peak_values['bins'] == 'below avg'].index.to_list(), sample_size)
    
    sample = np.zeros(int(sample_size*3))
    for i in range(0, sample_size):
        sample[0+i] = years[abv_avg_i[i]]
        sample[sample_size+i] = years[avg_i[i]]
        sample[int(sample_size*2)+i] = years[bel_avg_i[i]]

    return sample


## NASH-SUTCLIFFE EFFICIENCY ##
def nse(actual, predictions):
    return (1-(np.sum((actual-predictions)**2)/np.sum((actual-np.mean(actual))**2)))


## ANALYZE DATA ##
# for LSTM model
def analyze_results_lstm(model, metadata, test_swe, test_non_swe, scaler_swe, gen_plot, nonzero_swe):
    # # load files from model run - ADD THIS
    # model = torch.load(run_name+'_lstm.pt', map_location = DEVICE)
    # test_swe = torch.load(run_name + '_test_swe.pt')
    # test_non_swe = torch.load(run_name + '_test_non_swe.pt')
    # with open(run_name + '_normalize.pkl', 'rb') as file:  
    #     l_normalize = pickle.load(file)
    # scaler_swe = l_normalize_random[0]
    # test_metadata = pd.read_csv(run_name+'test_metadata.csv', sep=' ', index_col = 0)
    
    num_years = len(metadata)
    statistics = pd.DataFrame(0.0, index=np.arange(num_years), columns=['rmse', 'normal rmse', 'nse', 'r2', 'spearman_rho', 'delta peak', 
                                                                        'normal delta peak', 'abs delta peak', 'normal abs delta peak', 'delta days', 
                                                                        'abs delta days'])
    
    l_features = []
    
    ig = IntegratedGradients(model)
    
    if gen_plot: 
        plt.figure(figsize=(32,24))

    for i in range(0, num_years):
        site_id = metadata['site_id'][i]
        year = metadata['site_id'][i]
        test_swe_tensors = test_swe[i]
        test_non_swe_tensors = test_non_swe[i].to(DEVICE)
        test_non_swe_tensors = torch.reshape(test_non_swe_tensors, (test_non_swe_tensors.shape[0], 1, test_non_swe_tensors.shape[1]))
    
        # predict 
        swe_pred = model(test_non_swe_tensors)
    
        # calculate feature attribution
        attr, delta = ig.attribute(test_non_swe_tensors.requires_grad_(), return_convergence_delta=True)
        attr = attr.cpu().detach().numpy()
        l_features.append(np.mean(attr, axis=0)[0])
    
        # inverse transform to produce swe values
        swe_pred = scaler_swe.inverse_transform(swe_pred.cpu().detach().numpy().reshape(-1,1))
        swe_actual = scaler_swe.inverse_transform(test_swe_tensors.detach().numpy())

        # filter to only nonzero observational SWE values
        if nonzero_swe:
            zero_indexes = np.where(swe_actual != 0)[0]
            swe_actual = swe_actual[zero_indexes]
            swe_pred = swe_pred[zero_indexes]
    
        # peak swe
        peak_lstm = max(swe_pred)
        peak_obs = max(swe_actual)
        peak = (peak_lstm + peak_obs)/2
    
        # calculate RMSE
        mse = mean_squared_error(swe_actual, swe_pred)
        rmse = np.sqrt(mse)
        statistics.loc[i, 'rmse'] = rmse
        statistics.loc[i, 'normal rmse'] = rmse / peak
    
        # calculate NSE
        nash_sut = nse(swe_actual, swe_pred)
        statistics.loc[i, 'nse'] = nash_sut
    
        # calculate r2
        r_2 = r2_score(swe_actual, swe_pred)
        statistics.loc[i, 'r2'] = r_2
    
        # calculate Spearman's rho
        spearman_rho = stats.spearmanr(swe_actual, swe_pred)
        statistics.loc[i, 'spearman_rho'] = spearman_rho[0]
    
        # calculate regular and ABSOLUTE delta peak SWE
        statistics.loc[i, 'delta peak'] = peak_lstm - peak_obs
        statistics.loc[i, 'abs delta peak'] = np.abs(peak_lstm - peak_obs)
        statistics.loc[i, 'normal delta peak'] = (peak_lstm - peak_obs) / peak
        statistics.loc[i, 'normal abs delta peak'] = np.abs(peak_lstm - peak_obs) / peak
    
        # calculate first snow free day
        # obs/clm: swe == 0
        # pred: swe < 0
        arr_lstm = np.where(swe_pred < 0)[0]
        arr_obs = np.where(swe_actual == 0)[0]
        melt_lstm = np.where(arr_lstm > 100)[0]
        melt_obs = np.where(arr_obs > 100)[0]
        # calculate ABSOLUTE delta days
        if nonzero_swe:
            statistics.loc[i, 'delta days'] = 0
            statistics.loc[i, 'abs delta days'] = 0
        else: 
            try:
                statistics.loc[i, 'delta days'] = arr_lstm[melt_lstm[0]] - arr_obs[melt_obs[0]]
                statistics.loc[i, 'abs delta days'] = np.abs(arr_lstm[melt_lstm[0]] - arr_obs[melt_obs[0]])
            except:
                statistics.loc[i, 'delta days'] = 365 - arr_obs[melt_obs[0]]
                statistics.loc[i, 'abs delta days'] = np.abs(365 - arr_obs[melt_obs[0]])
    
        # plot first 36 years at 9 sites if boolean is true
        if gen_plot and (i < 36):
            plt.subplot(6, 6, i+1)
            # blue is actual, red is predicted
            plt.plot(swe_pred, label='predicted swe', c='red')
            plt.plot(swe_actual, label='actual swe', c='blue')
            plt.title(site_id + ': '+ str(year))
            #plt.title(f'{test_years['years'][y]:.0f}: RMSE: {rmse:.2f}') 
            plt.xlabel('days in WY')
            plt.ylabel('SWE [mm]')

    if gen_plot:
        plt.tight_layout()
    full = np.stack(l_features).astype(np.float32)

    return statistics, full


## PRINT AND VISUALIZE FEATURE IMPORTANCES ##
def visualize_importances(feature_names, importances, title="Average Feature Importances", plot=True, axis_title="Features"):
    print(title)
    for i in range(len(feature_names)):
        print(feature_names[i], ": ", '%.3f'%(importances[i]))
    x_pos = (np.arange(len(feature_names)))
    if plot:
        plt.figure(figsize=(12,6))
        plt.bar(x_pos, importances, align='center')
        plt.xticks(x_pos, feature_names, wrap=True)
        plt.xlabel(axis_title)
        plt.title(title)


## GET METRICS FOR  MODEL RUNS ##
# returns metrics for model runs given two lists of model names, stores them in folder model_type_output
def get_model_metrics(models, model_type):
    total_statistics = pd.DataFrame(columns=['run name','rmse', 'normal rmse', 'nse', 'r2', 'spearman_rho', 'delta peak', 'normal delta peak', 
                                             'abs delta peak', 'normal abs delta peak', 'delta days', 'abs delta days'])
    for run in models:
        statistics = pd.read_csv('Data/'+model_type+'output/'+run+'_statistics.txt',sep=' ',header=None)
        statistics.columns = ['rmse', 'normal rmse', 'nse', 'r2', 'spearman_rho', 'delta peak', 'normal delta peak', 'abs delta peak', 
                              'normal abs delta peak', 'delta days', 'abs delta days']
        total_statistics.loc[len(total_statistics)] = [run, np.mean(statistics['rmse']), np.mean(statistics['normal rmse']), np.mean(statistics['nse']), 
                                                       np.mean(statistics['r2']),np.mean(statistics['spearman_rho']), 
                                                       np.mean(statistics['delta peak']), np.mean(statistics['normal delta peak']), 
                                                       np.mean(statistics['abs delta peak']), np.mean(statistics['normal abs delta peak']),
                                                       np.mean(statistics['delta days']), np.mean(statistics['abs delta days'])]
    #total_statistics['normal delta days'] = total_statistics['delta days']/365
    #total_statistics['normal abs delta days'] = total_statistics['abs delta days']/365
    
    return total_statistics


## RUN LSTM MODEL FOR GIVEN YEAR ##
# runs model "model" for year defined by index
# returns actual and predicted SWE values for that year
def run_lstm(model, index):
    # load model components
    with open('/home/mcburns/national_lstm/Data/LSTM_output/'+model+'_normalize.pkl', 'rb') as file:  
        l_normalize = pickle.load(file)
    scaler_swe = l_normalize[0]
    ev_lstm = torch.load('Data/LSTM_output/'+model+'_lstm.pt', map_location = DEVICE)

    # get pre-loaded tensors from model output
    # test_swe_tensors = torch.load('Data/LSTM_output/'+model + '_test_swe.pt')
    # test_non_swe_tensors = torch.load('Data/LSTM_output/'+model + '_test_non_swe.pt')

    # create tensors from testing data and normalization function
    with open('/home/mcburns/national_lstm/Data/LSTM_output/test_swe.pkl', 'rb') as file:  
        test_swe = pickle.load(file)
    with open('/home/mcburns/national_lstm/Data/LSTM_output/test_non_swe.pkl', 'rb') as file:  
        test_non_swe = pickle.load(file)
    test_swe_tensors, test_non_swe_tensors, test_sites, test_years = create_dataset(test_swe, test_non_swe, l_normalize)

    # check if index is in range; if not, return empty arrays
    if(index > len(test_swe)):
        print('index ', index, ' is outside testing range of length ', len(test_swe))
        return [], []

    # run LSTM for given year (index)
    test_swe_input = test_swe_tensors[index]
    test_non_swe_input = test_non_swe_tensors[index].to(DEVICE)
    test_non_swe_input = torch.reshape(test_non_swe_input, (test_non_swe_input.shape[0], 1, test_non_swe_input.shape[1]))
    swe_pred = ev_lstm(test_non_swe_input)

    # inverse transform to produce SWE values in mm
    swe_pred = scaler_swe.inverse_transform(swe_pred.cpu().detach().numpy().reshape(-1,1))
    swe_actual = scaler_swe.inverse_transform(test_swe_input.cpu().detach().numpy().reshape(-1,1))

    return swe_pred, swe_actual


## RUN PARFLOW ##
# for given site_id and time period defined by start_date to end_date
def run_pf(site_id, start_date, end_date):
    # directories and run name
    os.environ["PARFLOW_DIR"] = "/home/SHARED/software/parflow/3.10.0"
    
    wy = str(pd.to_datetime(end_date).year)
    base = '/home/mcburns/national_lstm/Data/PFCLM_output/'+site_id+'/'+wy
    #print(base)
    PFCLM_SC = Run("pfclm")
    stopt = 8760                ## run for 365 days
    
    # File input version number
    PFCLM_SC.FileVersion = 4
    # Process Topology
    PFCLM_SC.Process.Topology.P = 1
    PFCLM_SC.Process.Topology.Q = 1
    PFCLM_SC.Process.Topology.R = 1
    # Computational Grid
    PFCLM_SC.ComputationalGrid.Lower.X = 0.0
    PFCLM_SC.ComputationalGrid.Lower.Y = 0.0
    PFCLM_SC.ComputationalGrid.Lower.Z = 0.0
    PFCLM_SC.ComputationalGrid.DX      = 2.0
    PFCLM_SC.ComputationalGrid.DY      = 2.0
    PFCLM_SC.ComputationalGrid.DZ      = 1.0
    PFCLM_SC.ComputationalGrid.NX      = 1
    PFCLM_SC.ComputationalGrid.NY      = 1
    PFCLM_SC.ComputationalGrid.NZ      = 10
    # The Names of the GeomInputs
    PFCLM_SC.GeomInput.Names = 'domain_input'
    # Domain Geometry Input
    PFCLM_SC.GeomInput.domain_input.InputType = 'Box'
    PFCLM_SC.GeomInput.domain_input.GeomName  = 'domain'
    # Domain Geometry
    PFCLM_SC.Geom.domain.Lower.X = 0.0
    PFCLM_SC.Geom.domain.Lower.Y = 0.0
    PFCLM_SC.Geom.domain.Lower.Z = 0.0
    PFCLM_SC.Geom.domain.Upper.X = 2.0
    PFCLM_SC.Geom.domain.Upper.Y = 2.0
    PFCLM_SC.Geom.domain.Upper.Z = 10.0
    PFCLM_SC.Geom.domain.Patches = 'x_lower x_upper y_lower y_upper z_lower z_upper'
    # variable dz assignments
    PFCLM_SC.Solver.Nonlinear.VariableDz = True
    PFCLM_SC.dzScale.GeomNames           = 'domain'
    PFCLM_SC.dzScale.Type                = 'nzList'
    PFCLM_SC.dzScale.nzListNumber        = 10
    # cells start at the bottom (0) and moves up to the top
    # domain is 8 m thick, root zone is down to 4 cells 
    # so the root zone is 2 m thick
    PFCLM_SC.Cell._0.dzScale.Value  = 1.0   
    PFCLM_SC.Cell._1.dzScale.Value  = 1.0   
    PFCLM_SC.Cell._2.dzScale.Value  = 1.0   
    PFCLM_SC.Cell._3.dzScale.Value  = 1.0
    PFCLM_SC.Cell._4.dzScale.Value  = 1.0
    PFCLM_SC.Cell._5.dzScale.Value  = 1.0
    PFCLM_SC.Cell._6.dzScale.Value  = 1.0
    PFCLM_SC.Cell._7.dzScale.Value  = 0.6   #0.6* 1.0 = 0.6  60 cm 3rd layer
    PFCLM_SC.Cell._8.dzScale.Value  = 0.3   #0.3* 1.0 = 0.3  30 cm 2nd layer
    PFCLM_SC.Cell._9.dzScale.Value  = 0.1   #0.1* 1.0 = 0.1  10 cm top layer
    # Perm
    PFCLM_SC.Geom.Perm.Names              = 'domain'
    PFCLM_SC.Geom.domain.Perm.Type        = 'Constant'
    PFCLM_SC.Geom.domain.Perm.Value       = 0.01465  # m/h 
    PFCLM_SC.Perm.TensorType              = 'TensorByGeom'
    PFCLM_SC.Geom.Perm.TensorByGeom.Names = 'domain'
    PFCLM_SC.Geom.domain.Perm.TensorValX  = 1.0
    PFCLM_SC.Geom.domain.Perm.TensorValY  = 1.0
    PFCLM_SC.Geom.domain.Perm.TensorValZ  = 1.0
    # Specific Storage
    PFCLM_SC.SpecificStorage.Type              = 'Constant'
    PFCLM_SC.SpecificStorage.GeomNames         = 'domain'
    PFCLM_SC.Geom.domain.SpecificStorage.Value = 1.0e-4
    # Phases
    PFCLM_SC.Phase.Names = 'water'
    PFCLM_SC.Phase.water.Density.Type     = 'Constant'
    PFCLM_SC.Phase.water.Density.Value    = 1.0
    PFCLM_SC.Phase.water.Viscosity.Type   = 'Constant'
    PFCLM_SC.Phase.water.Viscosity.Value  = 1.0
    # Contaminants
    PFCLM_SC.Contaminants.Names = ''
    # Gravity
    PFCLM_SC.Gravity = 1.0
    # Setup timing info
    PFCLM_SC.TimingInfo.BaseUnit     = 1.0
    PFCLM_SC.TimingInfo.StartCount   = 0
    PFCLM_SC.TimingInfo.StartTime    = 0.0
    PFCLM_SC.TimingInfo.StopTime     = stopt
    PFCLM_SC.TimingInfo.DumpInterval = 1.0
    PFCLM_SC.TimeStep.Type           = 'Constant'
    PFCLM_SC.TimeStep.Value          = 1.0
    # Porosity
    PFCLM_SC.Geom.Porosity.GeomNames    = 'domain'
    PFCLM_SC.Geom.domain.Porosity.Type  = 'Constant'
    PFCLM_SC.Geom.domain.Porosity.Value = 0.3
    # Domain
    PFCLM_SC.Domain.GeomName = 'domain'
    # Mobility
    PFCLM_SC.Phase.water.Mobility.Type  = 'Constant'
    PFCLM_SC.Phase.water.Mobility.Value = 1.0
    # Relative Permeability
    PFCLM_SC.Phase.RelPerm.Type        = 'VanGenuchten'
    PFCLM_SC.Phase.RelPerm.GeomNames   = 'domain'
    PFCLM_SC.Geom.domain.RelPerm.Alpha = 2.0
    PFCLM_SC.Geom.domain.RelPerm.N     = 3.0
    # Saturation
    PFCLM_SC.Phase.Saturation.Type        = 'VanGenuchten'
    PFCLM_SC.Phase.Saturation.GeomNames   = 'domain'
    PFCLM_SC.Geom.domain.Saturation.Alpha = 2.0
    PFCLM_SC.Geom.domain.Saturation.N     = 3.0
    PFCLM_SC.Geom.domain.Saturation.SRes  = 0.2
    PFCLM_SC.Geom.domain.Saturation.SSat  = 1.0
    # Wells
    PFCLM_SC.Wells.Names = ''
    
    # Time Cycles
    PFCLM_SC.Cycle.Names = 'constant'
    PFCLM_SC.Cycle.constant.Names = 'alltime'
    PFCLM_SC.Cycle.constant.alltime.Length = 1
    PFCLM_SC.Cycle.constant.Repeat = -1
    # Boundary Conditions: Pressure
    PFCLM_SC.BCPressure.PatchNames = 'x_lower x_upper y_lower y_upper z_lower z_upper'
    PFCLM_SC.Patch.x_lower.BCPressure.Type          = 'FluxConst'
    PFCLM_SC.Patch.x_lower.BCPressure.Cycle         = 'constant'
    PFCLM_SC.Patch.x_lower.BCPressure.alltime.Value = 0.0
    PFCLM_SC.Patch.y_lower.BCPressure.Type          = 'FluxConst'
    PFCLM_SC.Patch.y_lower.BCPressure.Cycle         = 'constant'
    PFCLM_SC.Patch.y_lower.BCPressure.alltime.Value = 0.0
    PFCLM_SC.Patch.z_lower.BCPressure.Type          = 'DirEquilRefPatch'
    PFCLM_SC.Patch.z_lower.BCPressure.RefGeom       = 'domain'
    PFCLM_SC.Patch.z_lower.BCPressure.RefPatch      = 'z_upper'
    PFCLM_SC.Patch.z_lower.BCPressure.Cycle         = 'constant'
    PFCLM_SC.Patch.z_lower.BCPressure.alltime.Value = -0.5 
    PFCLM_SC.Patch.x_upper.BCPressure.Type          = 'FluxConst'
    PFCLM_SC.Patch.x_upper.BCPressure.Cycle         = 'constant'
    PFCLM_SC.Patch.x_upper.BCPressure.alltime.Value = 0.0
    PFCLM_SC.Patch.y_upper.BCPressure.Type          = 'FluxConst'
    PFCLM_SC.Patch.y_upper.BCPressure.Cycle         = 'constant'
    PFCLM_SC.Patch.y_upper.BCPressure.alltime.Value = 0.0
    PFCLM_SC.Patch.z_upper.BCPressure.Type          = 'OverlandFlow'
    PFCLM_SC.Patch.z_upper.BCPressure.Cycle         = 'constant'
    PFCLM_SC.Patch.z_upper.BCPressure.alltime.Value = 0.0
    # Topo slopes in x-direction
    PFCLM_SC.TopoSlopesX.Type              = 'Constant'
    PFCLM_SC.TopoSlopesX.GeomNames         = 'domain'
    PFCLM_SC.TopoSlopesX.Geom.domain.Value = 0.1  #slope in X-direction to allow ponded water to run off
    # Topo slopes in y-direction
    PFCLM_SC.TopoSlopesY.Type              = 'Constant'
    PFCLM_SC.TopoSlopesY.GeomNames         = 'domain'
    PFCLM_SC.TopoSlopesY.Geom.domain.Value = 0.0
    # Mannings coefficient
    PFCLM_SC.Mannings.Type               = 'Constant'
    PFCLM_SC.Mannings.GeomNames          = 'domain'
    PFCLM_SC.Mannings.Geom.domain.Value  = 2.e-6
    # Phase sources:
    PFCLM_SC.PhaseSources.water.Type              = 'Constant'
    PFCLM_SC.PhaseSources.water.GeomNames         = 'domain'
    PFCLM_SC.PhaseSources.water.Geom.domain.Value = 0.0
    # Exact solution specification for error calculations
    PFCLM_SC.KnownSolution = 'NoKnownSolution'
    
    # Set solver parameters
    PFCLM_SC.Solver         = 'Richards'
    PFCLM_SC.Solver.MaxIter = 15000
    PFCLM_SC.Solver.Nonlinear.MaxIter           = 100
    PFCLM_SC.Solver.Nonlinear.ResidualTol       = 1e-5
    PFCLM_SC.Solver.Nonlinear.EtaChoice         = 'Walker1'
    PFCLM_SC.Solver.Nonlinear.EtaValue          = 0.01
    PFCLM_SC.Solver.Nonlinear.UseJacobian       = False
    PFCLM_SC.Solver.Nonlinear.DerivativeEpsilon = 1e-12
    PFCLM_SC.Solver.Nonlinear.StepTol           = 1e-30
    PFCLM_SC.Solver.Nonlinear.Globalization     = 'LineSearch'
    PFCLM_SC.Solver.Linear.KrylovDimension      = 100
    PFCLM_SC.Solver.Linear.MaxRestarts          = 5
    PFCLM_SC.Solver.Linear.Preconditioner       = 'PFMG'
    PFCLM_SC.Solver.PrintSubsurf                = False
    PFCLM_SC.Solver.Drop                        = 1E-20
    PFCLM_SC.Solver.AbsTol                      = 1E-9
    
    #Writing output options for ParFlow
    write_pfb = True  #only PFB output for water balance example
    #  PFB  no SILO
    PFCLM_SC.Solver.PrintSubsurfData         = False
    PFCLM_SC.Solver.PrintPressure            = False
    PFCLM_SC.Solver.PrintSaturation          = False
    PFCLM_SC.Solver.PrintCLM                 = write_pfb
    PFCLM_SC.Solver.PrintMask                = False
    PFCLM_SC.Solver.PrintSpecificStorage     = False
    PFCLM_SC.Solver.PrintEvapTrans           = False
    
    PFCLM_SC.Solver.WriteSiloMannings        = False
    PFCLM_SC.Solver.WriteSiloMask            = False
    PFCLM_SC.Solver.WriteSiloSlopes          = False
    PFCLM_SC.Solver.WriteSiloSaturation      = False
    
    #write output in NetCDF - CAN COMMENT OUT PRESSURE & SATURATION
    write_netcdf = False
    #PFCLM_SC.NetCDF.NumStepsPerFile          = 8760
    #PFCLM_SC.NetCDF.WritePressure            = write_netcdf  
    PFCLM_SC.NetCDF.WriteSubsurface          = False
    #PFCLM_SC.NetCDF.WriteSaturation          = write_netcdf
    PFCLM_SC.NetCDF.WriteCLM                 = write_netcdf
    #PFCLM_SC.NetCDF.CLMNumStepsPerFile       = 240
    
    # LSM / CLM options - set LSM options to CLM
    PFCLM_SC.Solver.LSM              = 'CLM'
    # specify type of forcing, file name and location
    PFCLM_SC.Solver.CLM.MetForcing   = '1D'
    PFCLM_SC.Solver.CLM.MetFileName = site_id + '_' + str(wy)+'_forcing.txt'
    PFCLM_SC.Solver.CLM.MetFilePath  = '.'
    
    # Set CLM Plant Water Use Parameters
    PFCLM_SC.Solver.CLM.EvapBeta       = 'Linear'
    PFCLM_SC.Solver.CLM.VegWaterStress = 'Saturation'
    PFCLM_SC.Solver.CLM.ResSat         = 0.25
    PFCLM_SC.Solver.CLM.WiltingPoint   = 0.25
    PFCLM_SC.Solver.CLM.FieldCapacity  = 1.0       
    PFCLM_SC.Solver.CLM.IrrigationType = 'none'
    PFCLM_SC.Solver.CLM.RootZoneNZ     =  3   # layer used for seasonal Temp for LAI
    PFCLM_SC.Solver.CLM.SoiLayer       =  4   # root zone thickness, see above
    
    #PFCLM_SC.Solver.CLM.UseSlopeAspect = True
    
    #Writing output options for CLM
    #  no SILO, no native CLM logs
    PFCLM_SC.Solver.PrintLSMSink        = False
    PFCLM_SC.Solver.CLM.CLMDumpInterval = 1
    PFCLM_SC.Solver.CLM.CLMFileDir      = base
    PFCLM_SC.Solver.CLM.BinaryOutDir    = False
    PFCLM_SC.Solver.CLM.IstepStart      = 1
    PFCLM_SC.Solver.WriteCLMBinary      = False
    PFCLM_SC.Solver.WriteSiloCLM        = False
    PFCLM_SC.Solver.CLM.WriteLogs       = False
    PFCLM_SC.Solver.CLM.WriteLastRST    = True
    PFCLM_SC.Solver.CLM.DailyRST        = False
    PFCLM_SC.Solver.CLM.SingleFile      = True
    

    # Initial conditions: water pressure
    PFCLM_SC.ICPressure.Type                 = 'HydroStaticPatch'
    PFCLM_SC.ICPressure.GeomNames            = 'domain'
    PFCLM_SC.Geom.domain.ICPressure.Value    = 2.00
    PFCLM_SC.Geom.domain.ICPressure.RefGeom  = 'domain'
    PFCLM_SC.Geom.domain.ICPressure.RefPatch = 'z_lower'
    
    # Run ParFlow prior to changes
    PFCLM_SC.run(working_directory=base)
    return


## GET SINGLE COLUMN DATA FOR PARFLOW ##
# writes all PFCLM files to site_id and wy folders, returns nothing
# assume start_date and end_date are water year adjusted
def get_sc_data_pf(site_id, start_date, end_date):
    # CREATE PATH
    wy = str(pd.to_datetime(end_date).year)
    path =  os.path.join('Data/PFCLM_output', site_id, str(wy))
    if not os.path.exists(path):
       os.makedirs(path)
        
    # GET SNOTEL DATA 
    site_df_swe = hf.get_point_data(dataset='snotel', variable='swe', temporal_resolution='daily', aggregation='sod', 
                                             date_start=start_date, date_end=end_date, site_ids=[site_id])
    site_df_precip = hf.get_point_data(dataset='snotel', variable='precipitation', temporal_resolution='daily', aggregation='sum', 
                                   date_start=start_date, date_end=end_date, site_ids=[site_id])
    site_df_temp = hf.get_point_data(dataset='snotel', variable='air_temp', temporal_resolution='daily', aggregation='mean', 
                                 date_start=start_date, date_end=end_date, site_ids=[site_id])
    metadata_df = hf.get_point_metadata(dataset='snotel', variable='swe', temporal_resolution='daily', aggregation='sod', 
                                             date_start=start_date, date_end=end_date, site_ids=[site_id])
        
    swe = site_df_swe.set_axis(['date','swe'], axis='columns')
    precip = site_df_precip.set_axis(['date','precip'], axis='columns')
    temp = site_df_temp.set_axis(['date','temp'], axis='columns')
    
    swe.to_csv(os.path.join(path, site_id + '_' + str(wy)+'_swe.txt'),sep=' ',header=None, index=False, index_label=False)
    print('loaded SNOTEL data')
    
    # PARAMETERS FOR CW3E DATA
    # adjust end date for CW3E
    end_date_cw3e = str(pd.to_datetime(end_date) + DateOffset(days=1))
    
    lat = metadata_df['latitude'][0]               	
    lon = metadata_df['longitude'][0]              
    bounds = st.latlon_to_ij([[lat, lon], [lat, lon]], "conus2")
    
    # GET CW3E DATA
    variables = ["precipitation", "downward_longwave", "downward_shortwave", "specific_humidity", "air_temp", "east_windspeed", "north_windspeed", 
                 "atmospheric_pressure"]
    if (not os.path.exists("output/"+site_id+"/"+str(wy)+"/hourly.nc")):
        hf.get_gridded_files({"dataset": "CW3E", "temporal_resolution": "hourly", "start_time": start_date, "end_time": end_date_cw3e, 
                              "grid_point": [bounds[0], bounds[1]]}, variables=variables, filename_template="output/"+site_id+"/"+str(wy)+"/hourly.nc")
    if (not os.path.exists("output/"+site_id+"/"+str(wy)+"/daily.nc")):
        hf.get_gridded_files({"dataset": "CW3E", "temporal_resolution": "daily", "start_time": start_date, "end_time": end_date_cw3e, 
                              "grid_point": [bounds[0], bounds[1]]}, variables=variables, filename_template="output/"+site_id+"/"+str(wy)+"/daily.nc")
    
    ds = xr.open_dataset('output/'+site_id+'/'+str(wy)+'/hourly.nc', engine='netcdf4', drop_variables=['y', 'x','latitude','longitude', 'Temp_min',
                                                                                                              'Temp_max'])
    df = ds.to_dataframe()
    met_data_hourly = pd.DataFrame({"DSWR":df['DSWR'].values, "DLWR":df['DLWR'].values, "precip":df['APCP'].values, "temp":df['Temp'].values, 
                             "wind (E)":df['UGRD'].values, "wind (N)":df['VGRD'].values,"pressure":df['Press'].values, "q":df['SPFH'].values})
    
    ds = xr.open_dataset("output/"+site_id+"/"+str(wy)+"/daily.nc", engine='netcdf4', drop_variables=['y', 'x','latitude','longitude', 'Temp_min',
                                                                                                             'Temp_max'])
    df = ds.to_dataframe()
    met_data_daily = pd.DataFrame({"DSWR":df['DSWR'].values, "DLWR":df['DLWR'].values, "precip":df['APCP'].values, "temp":df['Temp_mean'].values, 
                             "wind (E)":df['UGRD'].values, "wind (N)":df['VGRD'].values,"pressure":df['Press_atmos'].values, "q":df['SPFH'].values})
    
    print('loaded CW3E data')

    # PROCESS
    precip['precip'] = precip['precip'].fillna(0)
    temp['temp'] = temp['temp'].interpolate(method='linear', limit_direction='both')
    
    # BIAS CORRECTION
    # based on daily SNOTEL data, then apply to hourly
    test = temp['temp'] - (met_data_daily['temp'] - 273.15)
    temp_mean = test.mean()
    
    test = precip['precip'] - met_data_daily['precip']
    precip_mean = test.mean()
    
    met_data_hourly['temp'] = met_data_hourly['temp']+temp_mean
    met_data_hourly['precip'] = met_data_hourly['precip']+(precip_mean/86400)
    
    # SAVE DATA
    met_data_hourly.to_csv(os.path.join(path, site_id + '_' + str(wy)+'_forcing.txt'),sep=' ',header=None, index=False, index_label=False)
    
    # GET STATIC DATA
    static_filepaths = st.subset_static(bounds, dataset="conus2_domain", write_dir=path)
    clm_paths = st.config_clm(bounds, start=start_date, end=end_date, dataset="conus2_domain", write_dir=path)

    return


## GET SINGLE COLUMN METADATA FOR SITE ##
def get_metadata_pf(site_id, start_date, end_date):
    # preliminaries
    wy = str(pd.to_datetime(end_date).year)
    path =  os.path.join('Data/PFCLM_output', site_id, str(wy))
    if not os.path.exists(path):
       os.makedirs(path)

    # get snotel metadata
    metadata_snotel = hf.get_point_metadata(dataset='snotel', variable='swe', temporal_resolution='daily', aggregation='sod', 
                                        date_start=start_date, date_end=end_date, site_ids=[site_id])
    
    lat = metadata_snotel['latitude'][0]               	
    lon = metadata_snotel['longitude'][0]              
    bounds = st.latlon_to_ij([[lat, lon],[lat, lon]],"conus2")
    
    # get site metadata
    variables = ["veg_type_IGBP", "slope_x", "slope_y"]
    if (not os.path.exists(path+'/'+site_id+'_'+wy+'_static.nc')):
        hf.get_gridded_files({"dataset": "conus2_domain", "grid":"conus2", "start_time": start_date, "end_time": end_date, "grid_point": [bounds[0],
                                                                                                                                          bounds[1]]},
                             variables=variables, filename_template=path+"/"+site_id+"_{wy}_static.nc")
    
    ds = xr.open_dataset(path+'/'+site_id+'_'+wy+'_static.nc', engine='netcdf4', drop_variables=['y', 'x','latitude','longitude'])
    metadata_top = ds.to_dataframe()
    
    metadata = pd.DataFrame({"elevation":metadata_snotel['usda_elevation'][0],"latitude":metadata_snotel['latitude'][0],
                             "longitude":metadata_snotel['longitude'][0], "land_cover":metadata_top['vegetation_type'].values,
                             "slope_x":metadata_top['slope_x'].values, "slope_y":metadata_top['slope_y'].values}, index=[0])
    return metadata


## PRODUCE RESULTS ##
# returns PFCLM predictions and SNOTEL observational data for given site_id and year
def prod_swe(site_id, year):
    # read PFCLM output data
    try:
        clm_output = pd.read_csv(os.path.join('Data/PFCLM_output/'+site_id+'/'+str(year), 'clm_output.txt'),sep=' ',header=None, index_col=None)
    except:
        return
    clm_output.columns = ['LH [W/m2]', 'T [mm/s]', 'Ebs [mm/s]', 'Qflux infil [mm/s]', 'qflx_evap_tot [mm/s]', 'qflx_evap_grnd [mm/s]','SWE [mm]', 
                          'Tgrnd [K]']

    # adjust PFCLM output to daily resolution
    swe_clm = np.zeros(365)
    for i in range(0,365):
        i_hr = i*24
        avg = np.mean(clm_output['SWE [mm]'][i_hr:i_hr+24])
        swe_clm[i] = avg
    swe_clm = pd.DataFrame(swe_clm,columns=['swe'])
    swe_clm = swe_clm.interpolate(method='linear', limit_direction='both')

    # read actual SWE
    swe_actual = pd.read_csv(os.path.join('Data/PFCLM_output/'+site_id+'/'+str(year), site_id+'_'+str(year)+'_swe.txt'), sep=' ',header=None,index_col=False)
    swe_actual.columns = ['date', 'swe']

    if len(swe_actual) == 366:
        swe_actual = swe_actual.drop(60).reset_index()

    swe_actual['swe'] =  swe_actual['swe'].interpolate(method='linear', limit_direction='both')

    return swe_clm, swe_actual


## ANALYZE DATA ##
# for PFCLM and UA SWE model
def analyze_results_pfclm(swe_model, swe_actual, site_id, year, nonzero_swe):
    # filter to only nonzero observational SWE values if desired
    if nonzero_swe:
        zero_indexes = np.where(swe_actual != 0)[0]
        swe_actual = swe_actual[zero_indexes]
        swe_model = swe_model[zero_indexes]
            
    statistics = pd.DataFrame(columns=['rmse', 'normal rmse', 'nse', 'r2', 'spearman_rho', 'delta peak', 'normal delta peak', 'abs delta peak', 
                                       'normal abs delta peak', 'delta days', 'abs delta days'])
    
    # peak swe
    peak_model = max(swe_model)
    peak_obs = max(swe_actual)
    peak = (peak_model + peak_obs)/2

    # calculate RMSE
    rmse = np.sqrt(mean_squared_error(swe_actual, swe_model))

    # calculate NSE
    nash_sut = nse(swe_actual, swe_model)

    # calculate r2
    r_2 = r2_score(swe_actual, swe_model)

    # calculate Spearman's rho
    spearman_rho = stats.spearmanr(swe_actual, swe_model)

    # calculate regular and ABSOLUTE delta peak SWE
    delta_peak = peak_model - peak_obs
    abs_delta_peak = np.abs(peak_model - peak_obs)

    # calculate first snow free day
    # obs/clm: swe == 0
    # pred: swe < 0
    arr_lstm = np.where(swe_model == 0)[0]
    arr_obs = np.where(swe_actual == 0)[0]
    melt_lstm = np.where(arr_lstm > 100)[0]
    melt_obs = np.where(arr_obs > 100)[0]
    # calculate ABSOLUTE difference in first melt day
    if nonzero_swe:
        statistics.loc[i, 'delta days'] = 0
        statistics.loc[i, 'abs delta days'] = 0
    else:
        try:
            delta_days = arr_lstm[melt_lstm[0]] - arr_obs[melt_obs[0]]
            abs_delta_days = np.abs(arr_lstm[melt_lstm[0]] - arr_obs[melt_obs[0]])
        except:
            delta_days = 365 - arr_obs[melt_obs[0]]
            abs_delta_days = np.abs(365 - arr_obs[melt_obs[0]])

    statistics.loc[len(statistics)] = [rmse, rmse / peak, nash_sut, r_2, spearman_rho[0], delta_peak, delta_peak/peak, abs_delta_peak, 
                                       abs_delta_peak/peak, delta_days, abs_delta_days]
    
    return statistics

    
    
