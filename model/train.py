import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

#Load dataset
data=pd.read_csv('data/iris.csv')

#preprocess the data 
X=data.drop('species', axis=1)
y=data['species']

#split the data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Train the model
model= RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train,y_train)

#save the model

joblib.dump(model, 'model/iris_model.pkl')