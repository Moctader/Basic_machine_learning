import matplotlib as plt
import numpy as np
import pandas as pd
import seaborn as sns

titanic =pd.read_csv('C:/Users/golam/PycharmProjects/Cross_validation/titanic.csv')
titanic.head()
print(titanic.head())

# missing value
titanic.isnull().sum()
#print(titanic.isnull().sum())

# Missing value is filled with average value
titanic['Age'].fillna(titanic['Age'].mean(), inplace=True)
#print(titanic['Parch'].head(10))

# Combine Sibsp & parch

# for i, col in enumerate(['SibSip', 'Parch']):
#     plt.figure(i)
#     sns.catplot(x=col, y='Survived', data=titanic, kind='point', aspect=2, )

# Combine SibSp &  Parch
titanic['Family_cnt'] = titanic['SibSp'] + titanic['Parch']

titanic.drop(['PassengerId', 'SibSp', 'Parch'], axis=1, inplace=True)
print(titanic.head(5))
titanic.isnull().sum()

# Applying groupby method
titanic.groupby(titanic['Cabin'].isnull())['Survived'].mean()
titanic['Cabin_ind'] = np.where(titanic['Cabin'].isnull(), 0,1) # if missing value Then 0 else 1
print(titanic['Cabin_ind'].head(5))

# Identifying male and female by binary number
gender_num = {'male': 0, 'female': 1}
titanic['Sex'] = titanic['Sex'].map(gender_num) # Replaycing mal and female with 0 a nd 1

# Drop the unnecessary variable
titanic.drop(['Cabin', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)

# Cleaned dataset save for further use
titanic.to_csv('C:/Users/golam/PycharmProjects/Cross_validation/titanic_cleaned.csv')