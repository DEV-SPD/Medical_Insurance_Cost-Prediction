import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


df = pd.read_csv('insurance.csv')


# checking for null values
print(df.isnull().sum())

# prints top 5 rows of the dataframe
print(df.head())

# checking shape of dataframe
print(df.shape)

# finding a summary of dataset
print(df.describe())

# taking a count of no. of male and female taking insurance.
sns.countplot(x='sex', data=df)
plt.show()

# AGE DISTRIBUTION
sns.set()
plt.figure(figsize=(6, 6))
sns.distplot(df['age'])
plt.show()

# BMI distribution
sns.set()
plt.figure(figsize=(6, 6))
sns.distplot(df['bmi'])
plt.show()

# checking how many smokers are taking insurance
sns.countplot(x='smoker', data=df)
plt.show()

# No of dependents
sns.countplot(x='children', data=df)
plt.show()

# data preprocessing
df = df.replace({"sex": {'male': 1, 'female': 0}, "smoker": {'yes': 1, 'no': 0}})
df = df.replace({"region": {'southeast': 1, 'northwest': 0, 'southwest': 2, 'northeast': 3}})
print(df.head())

x = df.drop(columns='charges')
y = df.charges

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

model = LinearRegression()
model.fit(x, y)






