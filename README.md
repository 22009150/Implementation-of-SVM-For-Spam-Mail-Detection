# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary python packages using import statements.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Split the dataset using train_test_split.

4.Calculate Y_Pred and accuracy.

5.Print all the outputs.

6.End the Program. 
 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Archana.k
RegisterNumber:  212222240011
*/
import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')

data.head()

data.info()

data.isnull().sum()

x=data["EmailText"].values
y=data["Label"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extractiaon.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)

y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:

## Data.head()

![image](https://github.com/22009150/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118708624/fe0b7f7f-83fe-4a2f-92cf-6ff2d9fc10dc)

## Data.info()

![image](https://github.com/22009150/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118708624/12d98c81-9143-4867-bfa8-a21824dd384c)

## Data.innull().sum()

![image](https://github.com/22009150/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118708624/189a779a-8a86-4162-9097-190a187dedd5)

## y_pred:

![image](https://github.com/22009150/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118708624/15ee7319-a90a-419a-925a-630f98ad29cb)

## Accuracy:

![image](https://github.com/22009150/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118708624/b5ab93d7-ca77-4de7-9bf1-5194b0ab4666)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
