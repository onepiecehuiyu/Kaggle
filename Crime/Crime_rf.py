import time
import csv
import os

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

pd.options.mode.chained_assignment = None

goal = 'Category'
myid = 'Id'

# Load data
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

features_non_numeric = ['Dates', 'DayOfWeek', 'PdDistrict', 'Address']
features = ['Dates', 'hour', 'dark', 'DayOfWeek', 'PdDistrict', 'StreetNo', 'Address', 'X', 'Y']
train['StreetNo'] = train['Address'].apply(lambda x: x.split(' ', 1)[0] if x.split(' ', 1)[0].isdigit() else 0)
train['Address'] = train['Address'].apply(lambda x: x.split(' ', 1)[1] if x.split(' ', 1)[0].isdigit() else x)
test['StreetNo'] = test['Address'].apply(lambda x: x.split(' ', 1)[0] if x.split(' ', 1)[0].isdigit() else 0)
test['Address'] = test['Address'].apply(lambda x: x.split(' ', 1)[1] if x.split(' ', 1)[0].isdigit() else x)
train['hour'] = train['Dates'].apply(lambda x: x[11:13] if len(x) > 4 else 12)
train['dark'] = train['Dates'].apply(lambda x: 1 if (len(x) > 4 and x[11:13] >= 18 and x[11:13] < 6) else 0)
test['hour'] = test['Dates'].apply(lambda x: x[11:13] if len(x) > 4 else 12)
test['dark'] = test['Dates'].apply(lambda x: 1 if (len(x) > 4 and x[11:13] >= 18 and x[11:13] < 6) else 0)

# Pre-processing non-number values
le = LabelEncoder()
for col in features_non_numeric:
	le.fit(list(train[col]) + list(test[col]))
	train[col] = le.transform(train[col])
	test[col] = le.transform(test[col])

scaler = StandardScaler()
for col in features:
	scaler.fit(list(train[col]) + list(test[col]))
	train[col] = scaler.transform(train[col])
	test[col] = scaler.transform(test[col])

# Define classifiers
classifiers = [RandomForestClassifier(max_depth=6, n_estimators=50)]

count = 0
for classifier in classifiers:
	# Train
	print classifier.__class__.__name__
	start = time.time()
	classifier.fit(np.array(train[list(features)]), train[goal])
	print '  -> Training time:', time.time() - start

	# Export result
	count += 1
	if not os.path.exists('result/'):
		os.makedirs('result/')
	# test[myid] values will get converted to float since column_stack will result in array
	predictions = np.column_stack((test[myid], classifier.predict_proba(np.array(test[features])))).tolist()
	predictions = [[int(i[0])] + i[1:] for i in predictions]
	csvfile = 'result/' + classifier.__class__.__name__ + '-' + str(count) + '-submit.csv'
	with open(csvfile, 'w') as output:
		writer = csv.writer(output, lineterminator='\n')
		writer.writerow([myid, 'ARSON', 'ASSAULT', 'BAD CHECKS', 'BRIBERY', 'BURGLARY', 'DISORDERLY CONDUCT',
						 'DRIVING UNDER THE INFLUENCE', 'DRUG/NARCOTIC', 'DRUNKENNESS', 'EMBEZZLEMENT',
						 'EXTORTION', 'FAMILY OFFENSES', 'FORGERY/COUNTERFEITING', 'FRAUD', 'GAMBLING',
						 'KIDNAPPING', 'LARCENY/THEFT', 'LIQUOR LAWS', 'LOITERING', 'MISSING PERSON',
						 'NON-CRIMINAL', 'OTHER OFFENSES', 'PORNOGRAPHY/OBSCENE MAT', 'PROSTITUTION',
						 'RECOVERED VEHICLE', 'ROBBERY', 'RUNAWAY', 'SECONDARY CODES', 'SEX OFFENSES FORCIBLE',
						 'SEX OFFENSES NON FORCIBLE', 'STOLEN PROPERTY', 'SUICIDE', 'SUSPICIOUS OCC', 'TREA',
						 'TRESPASS', 'VANDALISM', 'VEHICLE THEFT', 'WARRANTS', 'WEAPON LAWS'])
		writer.writerows(predictions)
