## TRAIN AND EVALUATE LSTM MODEL ##

## PRELIMINARIES ##
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import random as rn
import pickle
import traceback
import torch

from _lstm import *
from _data import *

import io
from contextlib import redirect_stdout
trap = io.StringIO()

func_map_year = { 'get_years_precip' : get_years_precip, 'get_years_random' : get_years_random}
func_map_site = { 'get_sites_full' : get_sites_full, 'get_sites_random' : get_sites_random, 'get_sites_longitude' : get_sites_longitude, 
                 'get_sites_longitude_bypercent': get_sites_longitude_bypercent, 'get_sites_latitude' : get_sites_latitude}

## DEFINE RUN ##
# inputs are run name, site selection method, number of sites, year selection method, number of years
run = input("run name: ")
func_sites = input("function for sites (get_sites_full or get_sites_random or get_sites_longitude or get_sites_longitude_bypercent or get_sites_latitude): ") 
num_sites = int(input("number of train sites: ")) #*2                       # double for testing
func_years = input("function for years (get_years_precip or get_years_random): ") 
num_years = int(input("number of train years for each site: "))             # or for each precipitation bucket

## GET SNOTEL SITES ##
snotel = func_map_site[func_sites](num_sites, num_years)


## GET TRAINING/TESTING DATA ##
data = pd.DataFrame(columns=['site_id', 'year','train'])

for i in range(0, len(snotel)):
    site_id = snotel['site_id'][i]
    start_date = snotel['first_wy_date'][i]
    end_date = snotel['last_wy_date'][i]
    try:
        with redirect_stdout(trap):
            years = func_map_year[func_years](start_date, end_date, site_id, num_years) 

        for j in range(0, len(years)):
            # alternate sites for training and testing data
            #if ((i%2) == 0):
            data.loc[len(data.index)] = [site_id, int(years[j]), True] 
            #else:
                #data.loc[len(data.index)] = [site_id, int(years[j]), False] 
    except:
        print('missing data for ', site_id)
        traceback.print_exc()

# get data for each year in training and testing set
l_swe_train = [] 
l_non_swe_train = []
l_swe_test = [] 
l_non_swe_test = []

data_train = data.loc[(data.loc[:,'train'] == True)].reset_index().drop(columns=['index','train'])
data_test = pd.read_csv('national_test_years.txt', sep=' ',header=None)
data_test.columns = ['site_id',	'year',	'train']
#data_test = data.loc[(data.loc[:,'train'] == False)].reset_index().drop(columns=['index','train'])

## GET TRAINING DATA ##
for j in range(0,len(data_train)):
    site_id = data_train['site_id'][j]
    year = data_train['year'][j]
    start_date = str(year-1) + '-10-01'
    end_date = str(year) + '-09-30'
    try:
        with redirect_stdout(trap):
            swe, non_swe = get_sc_data(site_id, start_date, end_date)
        l_swe_train.append(swe)
        l_non_swe_train.append(non_swe)

        # add site data
        data_train.loc[j, 'latitude'] = non_swe['latitude'][0]
        data_train.loc[j, 'longitude'] = non_swe['longitude'][0]
        data_train.loc[j, 'elevation'] = non_swe['elevation'][0]
        data_train.loc[j, 'land cover'] = non_swe['land_cover'][0]
        data_train.loc[j, 'slope_x'] = non_swe['slope_x'][0]
        data_train.loc[j, 'slope_y'] = non_swe['slope_y'][0] 
    except:
        print('missing data for ', site_id, " : ", year)
        traceback.print_exc()

## GET TESTING DATA ##
for j in range(0, len(data_test)):
    site_id = data_test['site_id'][j]
    year = data_test['year'][j]
    start_date = str(year-1) + '-10-01'
    end_date = str(year) + '-09-30'
    try:
        with redirect_stdout(trap):
            swe, non_swe = get_sc_data(site_id, start_date, end_date)
        l_swe_test.append(swe)
        l_non_swe_test.append(non_swe)

        # add site data
        data_test.loc[j, 'latitude'] = non_swe.loc[0,'latitude']
        data_test.loc[j, 'longitude'] = non_swe.loc[0,'longitude']
        data_test.loc[j, 'elevation'] = non_swe.loc[0,'elevation']
        data_test.loc[j, 'land cover'] = non_swe.loc[0,'land_cover']
        data_test.loc[j, 'slope_x'] = non_swe.loc[0,'slope_x']
        data_test.loc[j, 'slope_y'] = non_swe.loc[0,'slope_y']
    except:
        print('missing data for ', site_id, " : ", year)
        traceback.print_exc()


## SAVE METADATA ##
data_train.to_csv('Data/LSTM_output/'+run+'_train_metadata.csv',sep=' ', index=False)
data_test.to_csv('Data/LSTM_output/'+run+'_test_metadata.csv',sep=' ', index=False)


## MODEL INPUT ##
train_swe = pd.concat(l_swe_train).reset_index().drop(columns='index')
train_non_swe = pd.concat(l_non_swe_train).reset_index().drop(columns='index')

test_swe = pd.concat(l_swe_test).reset_index().drop(columns='index')
test_non_swe = pd.concat(l_non_swe_test).reset_index().drop(columns='index')

# generate normalization, based on training data
l_normalize = create_normalization(train_swe, train_non_swe)
scaler_swe = l_normalize[0]
with open("Data/LSTM_output/"+run+"_normalize.pkl" , 'wb') as file:  
    pickle.dump(l_normalize, file)

# generate input for model
train_swe_tensors, train_non_swe_tensors, train_sites, train_years = create_dataset(train_swe, train_non_swe, l_normalize)
test_swe_tensors, test_non_swe_tensors, test_sites, test_years = create_dataset(test_swe, test_non_swe, l_normalize)
torch.save(train_swe_tensors, 'Data/LSTM_output/'+run+'_train_swe.pt')
torch.save(train_non_swe_tensors, 'Data/LSTM_output/'+run+'_train_non_swe.pt')
torch.save(test_swe_tensors, 'Data/LSTM_output/'+run+'_test_swe.pt')
torch.save(test_non_swe_tensors, 'Data/LSTM_output/'+run+'_test_non_swe.pt')


## ESTABLISH MODEL ##
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
lstm = LSTM(input_size, batch_size, output_size, hidden_size, num_layers)
lstm = lstm.to(DEVICE)
train_swe_tensors = train_swe_tensors.to(DEVICE)
train_non_swe_tensors = train_non_swe_tensors.to(DEVICE)
test_swe_tensors = test_swe_tensors.to(DEVICE)
test_non_swe_tensors = test_non_swe_tensors.to(DEVICE)


## TRAIN MODEL ##
loss_record = train_lstm(lstm, train_swe_tensors, train_non_swe_tensors, test_swe_tensors, test_non_swe_tensors)
# val_record


## SAVE MODEL TO PICKLE ##
torch.save(lstm, 'Data/LSTM_output/'+run+'_lstm.pt')


## EVALUATE LSTM ##
ev_lstm = torch.load('Data/LSTM_output/'+run+'_lstm.pt', map_location = DEVICE)
ev_test_swe = torch.load('Data/LSTM_output/'+run + '_test_swe.pt')
ev_test_non_swe = torch.load('Data/LSTM_output/'+run + '_test_non_swe.pt')

statistics, feature_importance = analyze_results_lstm(ev_lstm, data_test, ev_test_swe, ev_test_non_swe, scaler_swe, False)
print('statistics for: ' + run)
print(f"RMSE: {np.mean(statistics['rmse']):.2f}")
print(f"normal RMSE: {np.mean(statistics['normal rmse']):.2f}")
print(f"NSE: {np.mean(statistics['nse']):.2f}")
print(f"R2: {np.mean(statistics['r2']):.2f}")
print(f"Spearman's rho: {np.mean(statistics['spearman_rho']):.2f}")
print(f"delta peak SWE: {np.mean(statistics['delta peak']):.2f}")
print(f"normal delta peak SWE: {np.mean(statistics['normal delta peak']):.2f}")
print(f"absolute delta peak SWE: {np.mean(statistics['abs delta peak']):.2f}")
print(f"normal absolute delta peak SWE: {np.mean(statistics['normal abs delta peak']):.2f}")
print(f"delta days: {np.mean(statistics['delta days']):.2f}")
print(f"absolute delta days: {np.mean(statistics['abs delta days']):.2f}")

# save feature importance and statistics
feature_importance = pd.DataFrame(feature_importance)
feature_importance.to_csv('Data/LSTM_output/'+run+'_features.txt',sep=' ',header=None, index=False, index_label=False)
statistics.to_csv('Data/LSTM_output/'+run+'_statistics.txt',sep=' ',header=None, index=False, index_label=False)