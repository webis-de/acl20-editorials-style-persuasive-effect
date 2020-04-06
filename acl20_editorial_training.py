import itertools
import machine_learning
import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler
from functools import reduce #python 3

def get_features_map(df):
    cols_map = {
        'liwc': [l for l in df.columns if l.startswith('liwc_')],
        'nrc': [l for l in df.columns if l.startswith('nrc_')],
        'mpqa_arg': [l for l in df.columns if l.startswith('mpqa_arg_')],
        'mpqa_subjobg': [l for l in df.columns if l.startswith('mpqa_subjobg_')],
        'adu': [l for l in df.columns if l.startswith('adu_')],
        'lemma': [l for l in df.columns if l.startswith('lemma')],
    }
    if 'lemma' in cols_map['lemma']:
        cols_map['lemma'].remove('lemma')
    return cols_map

def get_all_feature_types_comb(df):
    cols_map = get_features_map(df)
    combs=[]
    for i in range(1, len(cols_map.keys())+1):
        combs.extend(list(itertools.combinations(cols_map.keys(), i)))
    return combs




def get_x_y(df, y, remove_outliers = True,  normalizing_method = "standard"):
    df = df.fillna(0)
    df_train = df[df['split_label'] == 'train']
    df_test = df[df['split_label'] == 'test']
    ## Remove outliers:
    cols = get_features_map(df_train).values()
    cols = reduce(lambda x,y: x+y,cols)
    if remove_outliers:
        print("removing outliers by clipping values...")
        
        #df_train_features = df_train[cols]
        df_train, df_test = machine_learning.clip_outliers(df_train, df_test, lower_percentile=1,  upper_percentile=99)
    
    ## X Y
    print("getting X y data...")
    X_train = df_train[cols]    
    y_train = df_train[y].values
    
    X_test = df_test[cols]
    y_test = df_test[y].values  
    
    ## Scale - normalize
    X_train, X_test = machine_learning.normalize(X_train, X_test, normalizing_method=normalizing_method)
        
    print('end of get_x_y.')
    return X_train, y_train, X_test, y_test

def get_instances_with_features_sub(X_train_df, X_test_df, features):
    cols_map = get_features_map(X_train_df)
    cols= []
    for f in features:
        cols.extend(cols_map[f])
    #print(cols)
    X_train = X_train_df[cols].values
    X_test = X_test_df[cols].values    
    
    return X_train, X_test
