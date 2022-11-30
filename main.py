#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 17:30:10 2022

@author: KRETSCHMANN, J-P. & LE FLOCH, G.

AIM: Create full dataframes, df_train and df_test.

NOTE: It uses the os package, which should NOT be run on the console for an optimal
run. Please run full cells or the full script, but NEVER evaluate on the command
line as it will yield an error.

If one wants to debug it, you can create a cell containing a single line and thus
run the corresponding cell.
"""

#%% Import packages

# System management
import os

# Data management
import pandas as pd
import pickle

#%% Import all dataframes

file_path = os.path.realpath(__file__)[:-7]+"data/" 
# WARNING: Don't run it on the command line.
list_files = os.listdir(file_path)

training_files = [x for x in list_files if x[:5] == 'learn']
test_files = [x for x in list_files if x[:4] == 'test']
description_files = [x for x in list_files if x[:4] not in ('test','lear')]

training_sets = {}
for i in range(len(training_files)):
     training_sets.update({i:pd.read_csv(f'{file_path}{training_files[i]}')})


test_sets = {}
for i in range(len(test_files)):
     test_sets.update({i:pd.read_csv(f'{file_path}{test_files[i]}')})
     
description_sets = {}
for i in range(len(description_files)):
     description_sets.update({f'{description_files[i][:-4]}':
                       pd.read_csv(f'{file_path}{description_files[i]}')})
     
#%% Merge all of them into df_train, df_test
     
df_train,df_test = training_sets[3],test_sets[3]

for i in range(3):
    df_train = df_train.merge(training_sets[i],on="PERSON_ID",
                        how="left")
    df_test = df_test.merge(test_sets[i],on="PERSON_ID",
                        how="left")

#%% Save the datasets into a .sav file

pickle.dump([df_train,df_test],open('data/datasets.sav', 'wb'))
