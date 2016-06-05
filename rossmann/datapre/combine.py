# -*- coding: UTF-8 -*-
import csv

import numpy
import pandas

__author__ = 'Whiker'
__mtime__ = '2016/6/2'

br = pandas.read_csv("../result/sub_BayesRidge.csv")
la = pandas.read_csv("../result/sub_lasso.csv")
rf = pandas.read_csv("../result/sub_rf.csv")
xg = pandas.read_csv("../result/sub_xgb_linear.csv")
sv = pandas.read_csv("../result/sub_svm.csv")

myId = br['Id'].values

b = br.iloc[:, 1].values
l = la.iloc[:, 1].values
r = rf.iloc[:, 1].values
x = xg.iloc[:, 1].values
s = sv.iloc[:, 1].values

data = numpy.divide(r + s, 2.0)
preds = numpy.column_stack((myId, data)).tolist()
preds = [[int(i[0])] + [float(i[1])] for i in preds]

with open("../result/sub_combine.csv", "w") as output:
	writer = csv.writer(output, lineterminator='\n')
	writer.writerow(["Id", "Sales"])
	writer.writerows(preds)
