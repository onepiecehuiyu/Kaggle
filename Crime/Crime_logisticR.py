__author__ = 'Whiker'
__mtime__ = '2016/5/18'

from datetime import datetime
import pandas, numpy, csv
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression


def address(data):
	it = [w for w in data.to_string().split() if w.isupper() and len(w) > 2]
	tr1 = it[0]
	tr2 = it[0]
	if len(it) != 1:
		tr2 = it[1]
	return tr1, tr2


def parse_time(x):
	DD = datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
	time = DD.hour  # *60+DD.minute
	day = DD.day
	month = DD.month
	year = DD.year
	return time, day, month, year


dataset = pandas.read_csv("train.csv")
labelData = dataset['Category'].values

# DayOfWeek PdDistrict
dataset = pandas.get_dummies(dataset['DayOfWeek']).join(dataset)
dataset = pandas.get_dummies(dataset['PdDistrict']).join(dataset)

# Address
# dataset['tr1'], dataset['tr2'] = zip(*dataset[['Address']].apply(func=address, axis=1))
# numpy.savetxt("AddressTostreet.csv", dataset[['tr1', 'tr2']], delimiter=',', fmt='%s')
# dataset = pandas.get_dummies(dataset['tr1']).join(dataset)
# dataset = pandas.get_dummies(dataset['tr2']).join(dataset)

# Dates
dataset['time'], dataset['day'], dataset['month'], dataset['year'] = zip(*dataset['Dates'].apply(func=parse_time))
dataset = pandas.get_dummies(dataset['time'], prefix='time').join(dataset)
dataset = pandas.get_dummies(dataset['day'], prefix='day').join(dataset)
dataset = pandas.get_dummies(dataset['month'], prefix='month').join(dataset)
dataset = pandas.get_dummies(dataset['year'], prefix='year').join(dataset)

# X Y
x = preprocessing.MinMaxScaler().fit_transform(dataset['X'].values)
dataset = pandas.DataFrame(data=x, columns=['xx']).join(dataset)
y = preprocessing.MinMaxScaler().fit_transform(dataset['Y'].values)
dataset = pandas.DataFrame(data=y, columns=['yy']).join(dataset)

dataset.drop(['DayOfWeek', 'PdDistrict', 'Address',
			  'X', 'Y',
			  # 'tr1', 'tr2',
			  'Dates', 'Resolution', 'Descript', 'Category'], inplace=True, axis=1)


testset = pandas.read_csv("test.csv")

# DayOfWeek PdDistrict
testset = pandas.get_dummies(testset['DayOfWeek']).join(testset)
testset = pandas.get_dummies(testset['PdDistrict']).join(testset)

# Address
# testset['tr1'], testset['tr2'] = zip(*testset[['Address']].apply(func=address, axis=1))
# testset = pandas.get_dummies(testset['tr1']).join(testset)
# testset = pandas.get_dummies(testset['tr2']).join(testset)

# Dates
testset['time'], testset['day'], testset['month'], testset['year'] = zip(*testset['Dates'].apply(func=parse_time))
testset = pandas.get_dummies(testset['time'], prefix='time').join(testset)
testset = pandas.get_dummies(testset['day'], prefix='day').join(testset)
testset = pandas.get_dummies(testset['month'], prefix='month').join(testset)
testset = pandas.get_dummies(testset['year'], prefix='year').join(testset)

# X Y
x = preprocessing.MinMaxScaler().fit_transform(testset['X'].values)
testset = pandas.DataFrame(data=x, columns=['xx']).join(testset)
y = preprocessing.MinMaxScaler().fit_transform(testset['Y'].values)
testset = pandas.DataFrame(data=y, columns=['yy']).join(testset)

testset.drop(['DayOfWeek', 'PdDistrict', 'Address',
			  'X', 'Y',
			  # 'tr1', 'tr2',
			  'Dates', 'Id'], inplace=True, axis=1)

# dataset = pandas.read_csv("train2.csv")
# testset = pandas.read_csv("test2.csv")
# trainData = dataset.iloc[:, :].values
# testData = testset.iloc[:, :].values

LRModel = LogisticRegression(penalty='l2', solver='newton-cg', multi_class='multinomial')
LRModel.fit(dataset.iloc[:, :].values, labelData)
print(LRModel.score(dataset.iloc[:, :].values, labelData))

# preds = LRModel.predict(testData)
# numpy.savetxt('submission_lr.csv', preds, delimiter=',', header='Category', comments='', fmt='%s')

# save answer
# cList = dataset2['Category'].unique().tolist()
# cList.insert(0, 'Id')
# csvRfile = file(unicode("submission_lr.csv", "utf8"), 'rb')
# reader = csv.reader(csvRfile)
# csvWfile = file(unicode("submission_lr2.csv", "utf8"), 'wb')
# writer = csv.writer(csvWfile)
# num = 0
# for line in reader:
# 	if num == 0:
# 		num = 1
# 		writer.writerow(cList)
# 		continue
# 	data = numpy.zeros((1, 39), dtype=int)[0]
# 	index = cList.index(line[0])
# 	data[index] = 1
# 	data = data.tolist()
# 	data.insert(0, num - 1)
# 	num += 1
# 	writer.writerow(data)
# csvRfile.close()
# csvWfile.close()
