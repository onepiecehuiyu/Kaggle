# -*- coding: UTF-8 -*-
import csv
from math import log, exp

import numpy
import pandas
from sklearn.ensemble import ExtraTreesRegressor

__author__ = 'Whiker'
__mtime__ = '2016/6/5'


def sale(data):
	data = int(data) + 1
	return log(data)


dataset = pandas.read_csv("input/train2_.csv")
testset = pandas.read_csv("input/test2_.csv")

dataset['Sale'] = dataset['Sales'].apply(sale)

labelData = dataset['Sale'].values
myId = testset['Id'].values

testset.drop(['Id'], inplace=True, axis=1)
testData = testset.iloc[:, :].values
dataset.drop(['Sales', 'Sale'], inplace=True, axis=1)
dataData = dataset.iloc[:, :].values

rfModel = ExtraTreesRegressor(n_estimators=100, n_jobs=4, random_state=2016)
rfModel.fit(dataData, labelData)
preds = numpy.column_stack((myId, rfModel.predict(testData))).tolist()
preds = [[int(i[0])] + [exp(float(i[1])) - 1] for i in preds]

with open("result/sub_et.csv", "w") as output:
	writer = csv.writer(output, lineterminator='\n')
	writer.writerow(["Id", "Sales"])
	writer.writerows(preds)
