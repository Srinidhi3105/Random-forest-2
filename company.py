import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


company = pd.read_csv("Company_Data.csv")
company.head()
company.info()
company.columns
company.shape

#converting all non numeric data into numeric
from sklearn.preprocessing import LabelEncoder

le_ShelveLoc = LabelEncoder()
le_Urban = LabelEncoder()
le_US = LabelEncoder()

company['n_ShelveLoc']= le_ShelveLoc.fit_transform(company['ShelveLoc'])
company['n_Urban'] = le_Urban.fit_transform(company['Urban'])
company['n_US'] = le_US.fit_transform(company['US'])

company = company.drop(['ShelveLoc','Urban','US'],axis=1)

#cnverting the target variable to categorical
company['Sales']=company['Sales'].astype('int')

#checking for null values
company.isnull().sum()

#histogram of continuous variable
num_bins=10
plt.hist(company['Sales'],num_bins)

#probability density function
sns.distplot(company['Sales'],bins=10)    

#plotting a boxplot
box = sns.boxplot(x='Sales',y='Urban',data=company)


#defining the target variable
Y= company['Sales']

#defining the independent variable 
X= company.drop(labels=['Sales'],axis=1)

#splitting the data into training and test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.4,random_state=20)

#model building
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=10,random_state=30)

model.fit(X_train,Y_train)

#prediction 
prediction_test = model.predict(X_test)
print(prediction_test)

#accuracy of the model
from sklearn import metrics
print("Accuracy= ",metrics.accuracy_score(Y_test,prediction_test))


