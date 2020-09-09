import matplotlib as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split



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


# Read the Cleaned titanic csv file for the train test split
titanic=pd.read_csv('C:/Users/golam/PycharmProjects/Cross_validation/titanic_cleaned.csv')
print(titanic.head(10))

# Dropping the survived from the main file for label (prediction)
features = titanic.drop(['Survived'], axis=1)
labels = titanic['Survived']

# Split the test train for the validation and independent test set
X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.4, random_state= 42)
X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.5, random_state= 42)

# Shows the percentges of the dataset

for dataset in (Y_train, Y_val, Y_test):
    print(round(len(dataset) / len(labels), 2))

#X_train.to_csv(C:/Users/golam/PycharmProjects/Cross_validation/train_features.csv', index = False)
X_train.to_csv('C:/Users/golam/PycharmProjects/Cross_validation/train_features.csv', index = False)
X_val.to_csv('C:/Users/golam/PycharmProjects/Cross_validation/val_features.csv', index = False)
X_test.to_csv('C:/Users/golam/PycharmProjects/Cross_validation/test_features.csv', index = False)


Y_train.to_csv('C:/Users/golam/PycharmProjects/Cross_validation/train_labels.csv', index = False)
Y_val.to_csv('C:/Users/golam/PycharmProjects/Cross_validation/val_labels.csv', index = False)
Y_test.to_csv('C:/Users/golam/PycharmProjects/Cross_validation/test_labels.csv', index = False)