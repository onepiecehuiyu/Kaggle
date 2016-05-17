titanic
============================================
python:
	pandas sklearn numpy
	==========================
method:
	random forest and xgboost
	==========================
feature:
	random forest:
		SibSp & Parch change to Family, Family > 1 = 1
		Age nan change to random(mean-std, mean+std)
		Sex age<=16 change to child, delete male
		Pclass delete classNum == 3
		Fare nan change to median
		Embarked delete S
	xgboost:
		Pclass, Sex, Embarked change to dummy variable
		Age nan set to mean
		Fare nan change to median
		SibSp plus Parch = Family
