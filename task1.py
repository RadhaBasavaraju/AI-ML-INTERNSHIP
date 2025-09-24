1.
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:/Users/radhi/Downloads/Titanic-Dataset.csv")
print(df.head())
print(df.info())
print(df.isnull().sum())

2.
df['Fare'].fillna(df['Fare'].mean(), inplace=True)
df['Age'].fillna(df['Age'].median(), inplace=True)


3.
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
print(df.head())

4.
num_features = ['Age', 'Fare', 'SibSp', 'Parch']
scaler = StandardScaler()
df[num_features] = scaler.fit_transform(df[num_features])
print(df.head())

5.
num_features = ['Age', 'Fare', 'SibSp', 'Parch']
for col in num_features:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

for col in num_features:
    df = remove_outliers_iqr(df, col)
print("Shape after removing outliers:", df.shape)
print(df[num_features].describe())
