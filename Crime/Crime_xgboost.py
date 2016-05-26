import csv
from datetime import datetime
import numpy

import pandas
from sklearn import preprocessing
import xgboost


def address(data):
    if data.find('/') != -1:
        return 1
    else:
        return 0


def parse_time(data):
    DD = datetime.strptime(data, "%Y-%m-%d %H:%M:%S")
    minute = DD.minute
    hour = DD.hour
    day = DD.day
    month = DD.month
    year = DD.year

    hour = abs(hour - 30)

    return minute, hour, day, month, year


dataset = pandas.read_csv("data\\train.csv")
testset = pandas.read_csv("data\\test.csv")

# label
labelset = pandas.DataFrame(data=dataset['Category'], columns=['Category'])
le = preprocessing.LabelEncoder()
le.fit(labelset['Category'].unique().tolist())
labelData = le.transform(labelset['Category'].values)
cList = le.classes_.tolist()
myId = testset['Id']

# Dates
dataset['minute'], dataset['hour'], dataset['day'], dataset['month'], dataset['year'] = zip(
    *dataset['Dates'].apply(func=parse_time))
testset['minute'], testset['hour'], testset['day'], testset['month'], testset['year'] = zip(
    *testset['Dates'].apply(func=parse_time))

# Address
dataset['Address'] = dataset['Address'].apply(func=address)
testset['Address'] = testset['Address'].apply(func=address)

# XY
x = preprocessing.scale(dataset['X'])
dataset = pandas.DataFrame(data=x, columns=['xx']).join(dataset)
y = preprocessing.scale(dataset['Y'])
dataset = pandas.DataFrame(data=y, columns=['yy']).join(dataset)
x = preprocessing.scale(testset['X'])
testset = pandas.DataFrame(data=x, columns=['xx']).join(testset)
y = preprocessing.scale(testset['Y'])
testset = pandas.DataFrame(data=y, columns=['yy']).join(testset)

# PdDistrict DayOfWeek
dataset = pandas.get_dummies(dataset['DayOfWeek']).join(dataset)
dataset = pandas.get_dummies(dataset['PdDistrict']).join(dataset)
testset = pandas.get_dummies(testset['DayOfWeek']).join(testset)
testset = pandas.get_dummies(testset['PdDistrict']).join(testset)

dataset.drop(['Dates', 'Descript', 'Resolution', 'Category',
              'X', 'Y', 'DayOfWeek', 'PdDistrict'], inplace=True, axis=1)
testset.drop(['Dates', 'X', 'Y', 'DayOfWeek', 'PdDistrict', 'Id'], inplace=True, axis=1)
trainData = dataset.iloc[:, :].values
testData = testset.iloc[:, :].values

offset = 600000
xgtrain = xgboost.DMatrix(trainData[:offset, :], label=labelData[:offset])
xgeval = xgboost.DMatrix(trainData[offset:, :], label=labelData[offset:])
xgtest = xgboost.DMatrix(testData)

params = {"booster": "gbtree", "objective": "multi:softprob", "num_class": 39, "max_delta_step": 1, "max_depth": 6}
watchlist = [(xgtrain, 'train'), (xgeval, 'val')]
model = xgboost.train(list(params.items()), xgtrain, 150, watchlist, early_stopping_rounds=100)
preds = numpy.column_stack((myId, model.predict(xgtest, ntree_limit=model.best_iteration))).tolist()
preds = [[int(i[0])] + i[1:] for i in preds]

cList.insert(0, 'Id')
with open("result/sub_xgb_softprob.csv", "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(cList)
    writer.writerows(preds)
