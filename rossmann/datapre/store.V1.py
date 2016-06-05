# -*- coding: UTF-8 -*-
import pandas
from sklearn import preprocessing


def COSYC(data):
	return 2015 - data


dfs = pandas.read_csv("../input/store.csv")

# StoreType/Assortment 变哑变量
dfs = pandas.get_dummies(dfs['StoreType'], prefix='StoreType').join(dfs)
dfs = pandas.get_dummies(dfs['Assortment'], prefix='Assortment').join(dfs)

# CompetitionDistance 填充0/标准化
dfs['CompetitionDistance'].fillna(0, inplace=True)
preprocessing.scale(dfs['CompetitionDistance'], copy=False)

# CompetitionOpenSinceYear 填充0/更改
dfs['CompetitionOpenSinceYear'].fillna(0, inplace=True)
dfs['COSYC'] = dfs['CompetitionOpenSinceYear'].apply(COSYC)

dfs.drop([
	'StoreType', 'Assortment', 'CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth', 'Promo2SinceWeek',
	'Promo2SinceYear', 'PromoInterval'],
	inplace=True, axis=1)

dfs.to_csv("../input/store2.csv", index=False)
