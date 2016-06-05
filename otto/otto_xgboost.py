import csv

import numpy
from sklearn import preprocessing
import xgboost

__author__ = 'Whiker'
__mtime__ = '2016/5/29'

import pandas

dataset = pandas.read_csv("input/train.csv")
testset = pandas.read_csv("input/test.csv")

# label change to int
labelData = dataset['target'].values
le = preprocessing.LabelEncoder()
le.fit(labelData)
cList = le.classes_.tolist()
labelData = le.transform(labelData)

myId = testset['id'].values

dataset.drop(['target', 'id'], inplace=True, axis=1)
testset.drop(['id'], inplace=True, axis=1)
dataset = dataset.iloc[:, :].values
testset = testset.iloc[:, :].values

offset = 50000

xgtrain = xgboost.DMatrix(data=dataset[:offset, :], label=labelData[:offset])
xgeval = xgboost.DMatrix(data=dataset[offset:, :], label=labelData[offset:])
xgtest = xgboost.DMatrix(data=testset)

params = {"booster": "gbtree", "objective": "multi:softprob", "num_class": 9, "max_delta_step": 1, "max_depth": 15}
watchlist = [(xgtrain, 'train'), (xgeval, 'val')]
model = xgboost.train(list(params.items()), xgtrain, 300, watchlist, early_stopping_rounds=100)
preds = numpy.column_stack((myId, model.predict(xgtest, ntree_limit=model.best_iteration))).tolist()
preds = [[int(i[0])] + i[1:] for i in preds]

cList.insert(0, 'id')
with open("result/sub_xgb_softprob.csv", "w") as output:
	writer = csv.writer(output, lineterminator='\n')
	writer.writerow(cList)
	writer.writerows(preds)
