titanic
============================================

###python package:<br>
* pandas sklearn numpy<br>


###method:<br>
* random forest and xgboost<br>

###feature:<br>
#####　random forest:<br>
> * SibSp & Parch change to Family, Family > 1 = 1<br>
> * Age nan change to random(mean-std, mean+std)<br>
> * Sex age<=16 change to child, delete male<br>
> * Pclass delete classNum == 3<br>
> * Fare nan change to median<br>
> * Embarked delete S<br>

#####　xgboost:<br>
> * Pclass, Sex, Embarked change to dummy variable<br>
> * Age nan set to mean<br>
> * Fare nan change to median<br>
> * SibSp plus Parch = Family<br>
