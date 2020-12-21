#import external libraries and functions
import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report , confusion_matrix

#load dataset from CSV file and show records
bank_data=pd.read_csv('')
bank_data.head()
bank_data.shape

user_data =pd.read_csv('')
user_data.head()
user_data.shape

#Check the dataset information
bank_data.info()
user_data.info()

#merge two csv files
datasets=pd.merge(bank_data,user_data, left_on='Collateral Value', right_on='Annual income', how='inner')
datasets.head()

#divide datasets into attribute and label
attr=datasets.drop('Annual income' , axis=1)
lbl=datasets['Annual income']

#split tha datasets
attr_train,attr_test,lbl_train,lbl_test = train_test_split(attr,lbl,test_size=0.20)

#train the data
classifier = DecisionTreeClassifier()
classifier.fit(attr_train , lbl_train)

#prediction

lbl_predict = classifier.predict(attr_test)
return lbl_predict
#print(lbl_predict)


#evaluation

mtrx = confusion_matrix(lbl_test,lbl_predict)
report = classification_report(lbl_test,lbl_predict)








