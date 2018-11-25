#!/usr/bin/env python

import calendar
import os

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

import config

train = pd.read_csv(os.path.join(config.orig_dataset_path, 'train.csv.zip'))
store = pd.read_csv(os.path.join(config.orig_dataset_path, 'store.csv.zip'))

print (train.shape, store.shape)

month_abbrs = calendar.month_abbr[1:]
month_abbrs[8] = 'Sept'


# 1) make integer Year,Month,Day columns instead of Date
# 2) join data from store table
def preprocess(df, stores):
    date = np.array([map(int, s.split('-')) for s in df['Date']])
    df = df.drop(['Date'], axis=1)
    df['Year'] = date[:, 0]
    df['Month'] = date[:, 1]
    df['Day'] = date[:, 2]
    df = df.join(stores, on='Store', rsuffix='_right')
    df = df.drop(['Store_right'], axis=1)

    print df.head()

    promo2_start_months = [(s.split(',') if not pd.isnull(s) else []) for s in df['PromoInterval']]

    for month_abbr in month_abbrs:
        df['Promo2Start_' + month_abbr] = np.array([(1 if month_abbr in s else 0) for s in promo2_start_months])
    df = df.drop(['PromoInterval'], axis=1)

    df = df.fillna(0)
    return df


train_prepared_fixed_date = preprocess(train, store)


def get_str_column_names(df):
    str_names = []
    for col in df.columns:
        for x in df[col]:
            if isinstance(x, str):
                str_names.append(col)
                break

    return str_names

train_inds = train_prepared_fixed_date[train_prepared_fixed_date['Year'] == 2014].index
test_inds = train_prepared_fixed_date[train_prepared_fixed_date['Year'] == 2015].index

print ('before iloc' , train_prepared_fixed_date.head())

train2 = train_prepared_fixed_date.iloc[train_inds]
test2 = train_prepared_fixed_date.iloc[test_inds]


str_cat_columns = get_str_column_names(train_prepared_fixed_date)

print str_cat_columns


# transform categorical columns with strings using LabelEncoder
def fix_strs(df, cat_names, test_df=None):
    df[cat_names] = df[cat_names].fillna(0)
    if test_df is not None:
        test_df[cat_names] = test_df[cat_names].fillna(0)
    for col in cat_names:
        enc = LabelEncoder()
        df[col] = enc.fit_transform(df[col])
        if test_df is not None:
            test_df[col] = enc.transform(test_df[col])
    return df, test_df


print ('before fix_strs' , train2.head())

train2, test2 = fix_strs(train2, str_cat_columns, test2)

print (train2.head())

all_cat_names = (['Store', 'DayOfWeek', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday',
                'StoreType', 'Assortment', 'Promo2']
                 + ['Promo2Start_' + month_abbr for month_abbr in month_abbrs])

if not os.path.exists(config.preprocessed_dataset_path):
    os.mkdir(config.preprocessed_dataset_path)

train2.to_csv(os.path.join(config.preprocessed_dataset_path, 'train'), sep='\t', header=False, index=False)
test2.to_csv(os.path.join(config.preprocessed_dataset_path, 'test'), sep='\t', header=False, index=False)

with open(os.path.join(config.preprocessed_dataset_path, 'cd'), 'w') as cd:
    for idx, name in enumerate(train2.columns):
        cd.write('{}\t{}\n'.format(
            idx,
            'Label' if name == 'Sales' else ('Categ\t' + name if name in all_cat_names else 'Num\t' + name))
        )

