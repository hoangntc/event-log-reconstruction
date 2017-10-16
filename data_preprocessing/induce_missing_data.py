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

if not os.path.isdir('../input/'):
    os.makedirs('../input/')
    
if not os.path.isdir(args.input_dir):
    os.makedirs(args.input_dir)
    
sys.path.insert(0, './../utils/')
from utils import *

#Load data

print('Loading data...')
print(args.name)

# Only consider Case, Activity, Timestamp
cols = ['CaseID', 'Activity', 'CompleteTimestamp']

# For Timestamp: Convert to time
if args.name == 'helpdesk':
    data = pd.read_csv(args.data_dir + args.data_file)
else:
    data = pd.read_csv(args.data_dir + args.data_file, usecols=['Case ID', 'Activity', 'Complete Timestamp'])
    data['Case ID'] = data['Case ID'].apply(lambda x: x.split(' ')[1])
    

# Format for each column     
data.columns = cols
data['CompleteTimestamp'] = pd.to_datetime(data['CompleteTimestamp'], errors='coerce')
data['CaseID'] = data['CaseID'].apply(pd.to_numeric)
data['Activity'] = data['Activity'].apply(str)


print('There are: {} cases'.format(len(data['CaseID'].unique())))
print('There are: {} activities'.format(len(data['Activity'].unique())))

print('-----Frequency of different activities-----')
print(data['Activity'].value_counts())

total_NA = int(data.shape[0]*(data.shape[1]-1)*args.nan_pct)
print('Number of missing values: {}'.format(total_NA))

# introduce missing Activities and Timestamps
missing_data = data.copy()
i = 0
while i < total_NA:
    row = np.random.randint(1, data.shape[0]) #exclude first row
    col = np.random.randint(1, data.shape[1]) #exclude CaseID
    if not pd.isnull(missing_data.iloc[row, col]): 
        missing_data.iloc[row, col] = np.nan
        i+=1
        
print('-----Frequency of different activities-----')
print(missing_data['Activity'].value_counts())

most_frequent_activity = missing_data['Activity'].value_counts().index[0]
print('Most frequent activity is: {}'.format(most_frequent_activity))


first_timestamp = missing_data['CompleteTimestamp'][0]


missing_df = calculateCumTimeInterval(missing_data)
missing_df['CumTimeInterval'] = missing_df['CumTimeInterval'].apply(convert2seconds)

df = calculateCumTimeInterval(data)
df['CumTimeInterval'] = df['CumTimeInterval'].apply(convert2seconds)


#Split data into train, val, test
print('Splitting complete data...')
groupByCase = df.groupby(['CaseID'])
missing_groupByCase = missing_df.groupby(['CaseID'])

# Split: 60% train, 20% validate, 20% test
train_size = int(len(groupByCase)*args.train_pct)
val_size = int(len(groupByCase)*args.val_pct)
test_size = len(groupByCase) - train_size - val_size


df_train = pd.DataFrame(columns=list(df))
df_val = pd.DataFrame(columns=list(df))
df_test = pd.DataFrame(columns=list(df))

for caseid, data_case in groupByCase:
    if caseid <= train_size:
        df_train = df_train.append(data_case)
    elif train_size < caseid <= (train_size+val_size):
        df_val = df_val.append(data_case)
    else:
        df_test = df_test.append(data_case)

        
print('Splitting missing data...')
        
missing_df_train = pd.DataFrame(columns=list(missing_df))
missing_df_val = pd.DataFrame(columns=list(missing_df))
missing_df_test = pd.DataFrame(columns=list(missing_df))

#Note: case start from 1 not 0
for caseid, data_case in missing_groupByCase:
    if caseid <= train_size:
        missing_df_train = missing_df_train.append(data_case)
    elif train_size < caseid <= train_size+val_size:
        missing_df_val = missing_df_val.append(data_case)
    else:
        missing_df_test = missing_df_test.append(data_case)
        
        
#get number of rows
print(df_train.shape, df_val.shape, df_test.shape)
train_row_num = df_train.shape[0]
val_row_num = df_val.shape[0]
test_row_num = df_test.shape[0]

avai_instance = 0
for row in range(len(missing_df_test)):
    if not pd.isnull(missing_df_test['CumTimeInterval'].iloc[row]) and not pd.isnull(missing_df_test['Activity'].iloc[row]):
        avai_instance+=1
        
print('Number of available row: {}'.format(avai_instance))

nan_instance = 0
for row in range(len(missing_df_test)):
    if pd.isnull(missing_df_test['CumTimeInterval'].iloc[row]) or pd.isnull(missing_df_test['Activity'].iloc[row]):
        nan_instance+=1
        
print('Number of nan row: {}'.format(nan_instance))


#Save dataframes
print('Saving dataframes...')
df_name = os.path.join(args.input_dir, 'complete_df_full_{}.csv'.format(args.nan_pct))
df.to_csv(df_name, index=False)

missing_df_name = os.path.join(args.input_dir, 'missing_df_full_{}.csv'.format(args.nan_pct))
missing_df.to_csv(missing_df_name, index=False)

#Save parameters
print('Saving parameters...')
file_name = os.path.join(args.input_dir, 'parameters_{}.pkl'.format(args.nan_pct))
with open(file_name, 'wb') as f: 
    pickle.dump(most_frequent_activity, f, protocol=2)
    pickle.dump(first_timestamp, f, protocol=2)
    pickle.dump(avai_instance, f, protocol=2)
    pickle.dump(nan_instance, f, protocol=2)
    pickle.dump(train_size, f, protocol=2)
    pickle.dump(val_size, f, protocol=2)
    pickle.dump(test_size, f, protocol=2)
    pickle.dump(train_row_num, f, protocol=2)
    pickle.dump(val_row_num, f, protocol=2)
    pickle.dump(test_row_num, f, protocol=2)
    
print('Finish!!!')