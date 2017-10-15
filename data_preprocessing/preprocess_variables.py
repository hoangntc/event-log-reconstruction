import os, sys
import argparse
import pandas as pd
import numpy as np
import pickle

from dateutil.parser import parse
from datetime import datetime
import time

pd.options.mode.chained_assignment = None #to run loop quicker without warnings

args_parser = argparse.ArgumentParser(description='induce missing data')
args_parser.add_argument('-n', '--name', required=True, metavar='<data_name>', help='Name of input file')
args_parser.add_argument('-d', '--data_dir', default='../data/', help='data dir')
args_parser.add_argument('--nan_pct', default=0.3, type=float, help='Nan percentage')
args_parser.add_argument('--train_pct', default=0.6, type=float, help='Train percentage')
args_parser.add_argument('--val_pct', default=0.2, type=float, help='Validate percentage')
args = args_parser.parse_args()
args.data_file = args.name + '.csv'
args.input_dir = '../input/{}/'.format(args.name)

#name = 'bpi_2012'
#name = 'bpi_2013'
#name = 'Road_Traffic_Fine_Management_Process'
'''
args = {
    'data_dir': '../data/',
    'data_file': name + '.csv',
    'input_dir': '../input/{}/'.format(name),  
    'nan_pct': 0.3,
    'train_pct': 0.6,
    'val_pct': 0.2,
}

args = argparse.Namespace(**args)
'''

file_name = os.path.join(args.input_dir, 'parameters_{}.pkl'.format(args.nan_pct))
with open(file_name, 'rb') as f:
    most_frequent_activity = pickle.load(f)
    first_timestamp = pickle.load(f)
    avai_instance = pickle.load(f)
    nan_instance = pickle.load(f)
    train_size = pickle.load(f)
    val_size = pickle.load(f)
    test_size = pickle.load(f)
    train_row_num = pickle.load(f)
    val_row_num = pickle.load(f)
    test_row_num = pickle.load(f)
    
sys.path.insert(0, './../utils/')
from utils import *


#Load data

complete_df_full_name = 'complete_df_full_{}.csv'.format(args.nan_pct)
missing_df_full_name = 'missing_df_full_{}.csv'.format(args.nan_pct)
print('Loading data:')
print(complete_df_full_name)
print(missing_df_full_name)

df_name = os.path.join(args.input_dir, complete_df_full_name)
df = pd.read_csv(df_name)

missing_df_name = os.path.join(args.input_dir, missing_df_full_name)
missing_df = pd.read_csv(missing_df_name)


#Preprocess data
print('Processing data...')
groupByCase = df.groupby(['CaseID'])
missing_groupByCase = missing_df.groupby(['CaseID'])

normalized_complete_df = pd.DataFrame(columns=list(df)+['NormalizedTime'])
normalized_missing_df = pd.DataFrame(columns=list(df)+['NormalizedTime'])
min_max_storage = {}

for i, j in zip(groupByCase, missing_groupByCase):
    temp, missing_temp, missing_case_storage = minmaxScaler(i[0], i[1], j[1])
    normalized_complete_df = normalized_complete_df.append(temp)
    normalized_missing_df = normalized_missing_df.append(missing_temp)
    min_max_storage.update(missing_case_storage)
    
normalized_complete_df_name = os.path.join(args.input_dir, 'normalized_complete_df_{}.csv'.format(args.nan_pct))
normalized_complete_df.to_csv(normalized_complete_df_name, index=False)

normalized_missing_df_name = os.path.join(args.input_dir, 'normalized_missing_df_{}.csv'.format(args.nan_pct))
normalized_missing_df.to_csv(normalized_missing_df_name, index=False)

cat_var = ['Activity']


# OHE: get k dummies out of k categorical levels (drop_first=False)
enc_complete_df = OHE(normalized_complete_df, cat_var)
enc_missing_df = OHE(normalized_missing_df, cat_var)

print('Getting masks...')

c_df = enc_complete_df.copy()
m_df = enc_missing_df.copy()
enc_complete_df_w_normalized_time = c_df.drop(['CompleteTimestamp', 'CumTimeInterval'], axis=1)
enc_missing_df_w_normalized_time = m_df.drop(['CompleteTimestamp', 'CumTimeInterval'], axis=1)

c_df = enc_complete_df.copy()
m_df = enc_missing_df.copy()
enc_complete_df_w_time = c_df.drop(['CompleteTimestamp', 'NormalizedTime'], axis=1)
enc_missing_df_w_time = m_df.drop(['CompleteTimestamp', 'NormalizedTime'], axis=1)

avai_index_df = enc_missing_df_w_time.copy()
nan_index_df = enc_missing_df_w_time.copy()

#mask for Time
print('Mask for Time')
for row in range(enc_missing_df_w_time.shape[0]):
    if np.isnan(enc_missing_df_w_time.loc[row, 'CumTimeInterval']): # if nan Time
        avai_index_df.loc[row, 'CumTimeInterval'] = 0
        nan_index_df.loc[row, 'CumTimeInterval'] = 1
    else:
        avai_index_df.loc[row, 'CumTimeInterval'] = 1
        nan_index_df.loc[row, 'CumTimeInterval'] = 0
        
#mask for Activity
print('Mask for Activity')
for row in range(enc_missing_df_w_time.shape[0]):
    if np.any(enc_missing_df_w_time.iloc[row,2:]>0): #if avai Time
        avai_index_df.iloc[row, 2:] = 1
        nan_index_df.iloc[row, 2:] = 0
    else:
        avai_index_df.iloc[row, 2:] = 0
        nan_index_df.iloc[row, 2:] = 1
        
pad_index_df = enc_complete_df.copy()
cols = [x for x in list(pad_index_df) if x != 'CaseID']
pad_index_df.loc[:, cols] = 1

enc_missing_df_w_normalized_time.fillna(0, inplace=True)
enc_missing_df_w_time.fillna(0, inplace=True)

enc_complete_w_normalized_time_groupByCase = enc_complete_df_w_normalized_time.groupby(['CaseID'])
enc_missing_w_normalized_time_groupByCase = enc_missing_df_w_normalized_time.groupby(['CaseID'])

enc_complete_w_time_groupByCase = enc_complete_df_w_time.groupby(['CaseID'])
enc_missing_w_time_groupByCase = enc_missing_df_w_time.groupby(['CaseID'])

avai_index_df_groupByCase = avai_index_df.groupby(['CaseID'])
nan_index_df_groupByCase = nan_index_df.groupby(['CaseID'])
pad_index_df_groupByCase = pad_index_df.groupby(['CaseID'])

maxlen = findLongestLength(groupByCase)
print('Length of longest case: {}'.format(maxlen))

cols_w_time = [i for i in list(enc_complete_df_w_time) if i != 'CaseID']
cols_w_normalized_time = [i for i in list(enc_complete_df_w_normalized_time) if i != 'CaseID']

vectorized_complete_df_w_normalized_time = getInput(enc_complete_w_normalized_time_groupByCase, cols_w_normalized_time, maxlen)
vectorized_missing_df_w_normalized_time = getInput(enc_missing_w_normalized_time_groupByCase, cols_w_normalized_time, maxlen)

vectorized_complete_df_w_time = getInput(enc_complete_w_time_groupByCase, cols_w_time, maxlen)
vectorized_missing_df_w_time = getInput(enc_missing_w_time_groupByCase, cols_w_time, maxlen)

vectorized_avai_index_df = getInput(avai_index_df_groupByCase, cols_w_time, maxlen)
vectorized_nan_index_df = getInput(nan_index_df_groupByCase, cols_w_time, maxlen)
vectorized_pad_index_df = getInput(pad_index_df_groupByCase, cols_w_time, maxlen)

# Split: 60% train, 20% validate, 20% test
print('Spitting data...')
#train
complete_matrix_w_normalized_time_train = vectorized_complete_df_w_normalized_time[:train_size]
missing_matrix_w_normalized_time_train = vectorized_missing_df_w_normalized_time[:train_size]

avai_matrix_train = vectorized_avai_index_df[:train_size]
nan_matrix_train = vectorized_nan_index_df[:train_size]

#validate
complete_matrix_w_normalized_time_val = vectorized_complete_df_w_normalized_time[train_size:train_size+val_size]
missing_matrix_w_normalized_time_val = vectorized_missing_df_w_normalized_time[train_size:train_size+val_size]

avai_matrix_val = vectorized_avai_index_df[train_size:train_size+val_size]
nan_matrix_val = vectorized_nan_index_df[train_size:train_size+val_size]
pad_matrix_val = vectorized_pad_index_df[train_size:train_size+val_size]

#test
complete_matrix_w_normalized_time_test = vectorized_complete_df_w_normalized_time[train_size+val_size:]
missing_matrix_w_normalized_time_test = vectorized_missing_df_w_normalized_time[train_size+val_size:]

#check number of avai instances in test set
print('Checking number of available instances in test set:...')
check_avai = avai_matrix_test.copy()
check_avai = check_avai.reshape(avai_matrix_test.shape[0]*avai_matrix_test.shape[1], avai_matrix_test.shape[2])
check_avai = check_avai[np.all(check_avai == 1, axis=1)]
print('Number of available row: {}'.format(check_avai.shape[0]))
print(check_avai.shape[0] == avai_instance)

print('\n')

#check number of nan instances in test set
print('Checking number of nan instances in test set:...')
check_nan = nan_matrix_test.copy()
check_nan = check_nan.reshape(nan_matrix_test.shape[0]*nan_matrix_test.shape[1], nan_matrix_test.shape[2])
check_nan = check_nan[np.any(check_nan != 0, axis=1)]
print('Number of nan row: {}'.format(check_nan.shape[0]))
print(check_nan.shape[0] == nan_instance)

avai_matrix_test = vectorized_avai_index_df[train_size+val_size:]
nan_matrix_test = vectorized_nan_index_df[train_size+val_size:]
pad_matrix_test = vectorized_pad_index_df[train_size+val_size:]


print('Saving preprocessed data...')
preprocessed_data_name = os.path.join(args.input_dir, 'preprocessed_data_{}.pkl'.format(args.nan_pct))
with open(preprocessed_data_name, 'wb') as f:
    pickle.dump(min_max_storage, f, protocol=2)
    pickle.dump(complete_matrix_w_normalized_time_train, f, protocol=2)
    pickle.dump(missing_matrix_w_normalized_time_train, f, protocol=2)
    pickle.dump(avai_matrix_train, f, protocol=2)
    pickle.dump(nan_matrix_train, f, protocol=2)
    pickle.dump(complete_matrix_w_normalized_time_val, f, protocol=2)
    pickle.dump(missing_matrix_w_normalized_time_val, f, protocol=2)
    pickle.dump(avai_matrix_val, f, protocol=2)
    pickle.dump(nan_matrix_val, f, protocol=2)
    pickle.dump(pad_matrix_val, f, protocol=2)
    pickle.dump(complete_matrix_w_normalized_time_test, f, protocol=2)
    pickle.dump(missing_matrix_w_normalized_time_test, f, protocol=2)
    pickle.dump(avai_matrix_test, f, protocol=2)
    pickle.dump(nan_matrix_test, f, protocol=2)
    pickle.dump(pad_matrix_test, f, protocol=2)
    pickle.dump(cols_w_time, f, protocol=2)
    pickle.dump(cols_w_normalized_time, f, protocol=2)
    
print('Finish!!!')