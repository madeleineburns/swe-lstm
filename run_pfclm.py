## RUN PFCLM MODEL ##

## PRELIMINARIES ##
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import io
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import random as rn
import traceback
from glob import glob
from tqdm import tqdm

from _pf_data import *

from contextlib import redirect_stdout
trap = io.StringIO()

## GET TESTING DATA and RUN PARFLOW ##
# read in testing data - 
data_test = pd.read_csv('/home/mcburns/national_lstm/national_test_years.txt', sep=' ',header=None)
data_test.columns = ['site_id',	'year',	'train']

total_statistics = pd.DataFrame(columns=['rmse', 'normal rmse', 'nse', 'r2', 'spearman_rho', 'delta peak', 'normal delta peak', 'abs delta peak',
                                         'normal abs delta peak','delta days', 'abs delta days'])

## RUN PARFLOW ##
# for each testing site
count_missing = 0
for j in tqdm(range(0, len(data_test))):
    #print('test: ', j)
    site_id = data_test['site_id'][j]
    year = data_test['year'][j]
    #print(site_id, " : ", year)
    start_date = str(year-1) + '-10-01'
    end_date = str(year) + '-09-30'

    # only run pfclm if it hasn't been run before for given site/year
    #if not os.path.exists(os.path.join('Data/PFCLM_output/'+site_id+'/'+str(year), 'clm_output.txt')):
    try:
        with redirect_stdout(trap):
            get_sc_data_pf(site_id, start_date, end_date)
            metadata = get_metadata_pf(site_id, start_date, end_date)
            run_pf(site_id, start_date, end_date)
    
        # create and save dataframe for CLM output
        files = glob('Data/PFCLM_output/'+site_id+'/'+str(year)+'/pfclm.out.clm_output.*.C.pfb')
        CLM_data = pf.read_pfb_sequence(files)
    
        # create data frame for CLM output & save it
        clm_output = pd.DataFrame({'LH [W/m2]':CLM_data[:,0,0,0],'T [mm/s]':CLM_data[:,8,0,0],'Ebs [mm/s]':CLM_data[:,6,0,0],
                                   'Qflux infil [mm/s]':CLM_data[:,9,0,0],'qflx_evap_tot [mm/s]':CLM_data[:,4,0,0],
                                   'qflx_evap_grnd [mm/s]':CLM_data[:,5,0,0], 'SWE [mm]':CLM_data[:,10,0,0],'Tgrnd [K]':CLM_data[:,11,0,0] })
        clm_output.to_csv(os.path.join('Data/PFCLM_output/'+site_id+'/'+str(year), 'clm_output.txt'),sep=' ',header=None, index=False,
                          index_label=False)
    
        # remove CLM output files - might have to be glob.files
        for f in files:
            os.remove(f)
        for f in glob('Data/PFCLM_output/'+site_id+'/'+str(year)+'/pfclm.out.clm_output.*.C.pfb.dist'):
            os.remove(f)
    
        # add site data
        data_test.loc[j, 'latitude'] = metadata.loc[0,'latitude']
        data_test.loc[j, 'longitude'] = metadata.loc[0,'longitude']
        data_test.loc[j, 'elevation'] = metadata.loc[0,'elevation']
        data_test.loc[j, 'land cover'] = metadata.loc[0,'land_cover']
        data_test.loc[j, 'slope_x'] = metadata.loc[0,'slope_x']
        data_test.loc[j, 'slope_y'] = metadata.loc[0,'slope_y']
        
    except:
        #print('missing data for ', site_id, " : ", year)
        #traceback.print_exc()
        count_missing += 1

    # run statistics, add to statistics file
    swe_clm, swe_actual = prod_swe(site_id, year)
    single_statistics = analyze_results_pfclm(swe_clm['swe'], swe_actual['swe'], site_id, year)
    total_statistics = pd.concat([total_statistics, single_statistics], ignore_index=True)
        
print('number of failed PFCLM runs: ',count_missing,' out of ',len(data_test),' sites')

total_statistics.to_csv('national_statistics.txt', sep=' ', header=None, index=False, index_label=False)

## SAVE METADATA 
#data_train.to_csv(run+'_train_metadata.csv',sep=' ', index=False)
#data_test.to_csv('Data/PFCLM_output/'+run+'_test_metadata.csv',sep=' ', index=False)

