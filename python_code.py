# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import pickle
data_train= pd.read_csv("train_data.csv")
data_test= pd.read_csv("test_data.csv")
claim= data_train["Claim"]
customer_id = data_test["Customer Id"]
data_train=data_train.drop('Claim', axis=1)
data = data_train.append(data_test)
data=data.drop("Customer Id", axis=1)
data=data.drop("NumberOfWindows", axis=1)
data.Garden = data.Garden.map({'V': 1, 'O': 0})
data.Building_Painted= data.Building_Painted.map({'V': 0, 'N': 1})
data.Building_Fenced= data.Building_Fenced.map({'V': 0, 'N': 1})
data.Settlement= data.Settlement.map({'U': 2, 'R': 0})
data.Garden.fillna(np.random.randint(0,2), inplace = True)
data.Building_Dimension.fillna(data.Building_Dimension.median(), inplace=True)
data.Date_of_Occupancy.fillna(data.Date_of_Occupancy.median(), inplace=True)
data["Building_Dimension"].values[data["Building_Dimension"].values < 30] =1808
data["Geo_Code"]= pd.to_numeric(data["Geo_Code"], errors='coerce')
data.Geo_Code.fillna(data.Geo_Code.median(), inplace = True)


from catboost import CatBoostClassifier
classifier= CatBoostClassifier(learning_rate = 0.005)
x_train=data.iloc[:7160,]
x_test= data.iloc[7160:,]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(x_train,claim,test_size=0.1)
classifier.fit(X_train, Y_train)


pickle.dump(classifier,open("model.pkl","wb")) 
model=pickle.load(open("model.pkl","rb"))
