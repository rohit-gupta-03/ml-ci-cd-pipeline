import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#load the data set
data= pd.read_csv('data/iris.csv')

#preprocess the data 
X=data.drop('species', axis=1)
y=data['species']

#split the data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#save the model
 model= joblib.load('model/iris_model.pkl')

#make the predications

y_pred= predict(X_test)


#Eval
accuracy=accuracy_score(y_test,y_pred)
print(f' Model accuracy:{accuracy:.2f}')