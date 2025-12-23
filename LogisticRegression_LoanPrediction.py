import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"D:\Internship\loanfile.csv")
df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].mean())
df["Credit_History"] = df["Credit_History"].fillna(df["Credit_History"].median())
df.dropna(inplace = True)

plt.figure()
plt.subplot(2,3,1)
sns.countplot(x = df['Gender'], hue = df['Loan_Status'], palette = 'Pastel1')

plt.subplot(2,3,2)
sns.countplot(x = df['Married'], hue = df['Loan_Status'], palette = 'Purples')

plt.subplot(2,3,3)
sns.countplot(x = df['Education'], hue = df['Loan_Status'], palette = 'Wistia')

plt.subplot(2,3,4)
sns.countplot(x = df['Self_Employed'], hue = df['Loan_Status'], palette = 'autumn')

plt.subplot(2,3,5)
sns.countplot(x = df['Property_Area'], hue = df['Loan_Status'], palette = 'gist_rainbow')

plt.subplot(2,3,6)
sns.countplot(x = df['Dependents'], hue = df['Loan_Status'], palette = 'Spectral')

plt.show()

df['Loan_Status'].map({'Y':1,'N':0})

df.Gender = df.Gender.map({'Male': 1, 'Female': 0})
df.Married = df.Married.map({'Yes': 1, 'No': 0})
df.Dependents = df.Dependents.map({'0': 0, '1': 1, '2': 2, '3+': 3})
df.Self_Employed= df.Self_Employed.map({'Yes': 1, 'No': 0})
df.Education = df.Education.map({'Graduate': 1, 'Non Graduate': 0})
df.Property_Area = df.Property_Area.map({'Urban': 1, 'Rural': 0})

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
df.dropna(inplace = True)
X = df.iloc[1:542, 1:11].values
Y = df.iloc[1:542, 12].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)
model = LogisticRegression()
model.fit(X_train, Y_train)

lr_Prediction = model.predict(X_test)
print("LR ACCURACY = ", metrics.accuracy_score(lr_Prediction, Y_test))
print()
print("Y_Predicted Values", lr_Prediction)
print()
print("Y_test", Y_test)

