import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
import joblib


def train_titanic():
    base_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    data_file = os.path.join(base_dir, 'data/titanic.csv') 
    
    df = pd.read_csv(data_file)
    df['Embarked'].fillna(value='S', inplace=True)

    mean_age_miss = df[df["Name"].str.contains('Miss.', na=False)]['Age'].mean().round()
    mean_age_mrs = df[df["Name"].str.contains('Mrs.', na=False)]['Age'].mean().round()
    mean_age_mr = df[df["Name"].str.contains('Mr.', na=False)]['Age'].mean().round()
    mean_age_master = df[df["Name"].str.contains('Master.', na=False)]['Age'].mean().round()

    df['Age'] = df[['Name', 'Age']].apply(fill_age, axis=1, args=(mean_age_miss, mean_age_mrs, mean_age_mr, mean_age_master))
    df['Cabin'] = pd.Series(['X' if pd.isnull(ii) else ii[0] for ii in df['Cabin']])
    df['Cabin'] = df[['Cabin', 'Fare']].apply(reasign_cabin, axis=1)
    df['Alone'] = df[['SibSp','Parch']].apply(create_alone_feature, axis=1)
    df['Familiars'] = 1 + df['SibSp'] + df['Parch']
    
    categories = {"female": 1, "male": 0}
    df['Sex']= df['Sex'].map(categories)

    categories = {"S": 1, "C": 2, "Q": 3}
    df['Embarked']= df['Embarked'].map(categories)

    categories = df.Cabin.unique()
    df['Cabin'] = df.Cabin.astype("category").cat.codes
    df = df.drop(['Name','Ticket','PassengerId'], axis=1)
    
    x = df[df.columns.difference(['Survived'])]
    y = df['Survived']

    classifier = RandomForestClassifier()
    classifier.fit(x, y)

    return classifier
    


def fill_age(
    name_age, 
    mean_age_miss, 
    mean_age_mrs, 
    mean_age_mr, 
    mean_age_master):
    name = name_age[0]
    age = name_age[1]
    
    if pd.isnull(age):
        if 'Mr.' in name:
            return mean_age_mr
        if 'Mrs.' in name:
            return mean_age_mrs
        if 'Miss.' in name:
            return mean_age_miss
        if 'Master.' in name:
            return mean_age_master
        if 'Dr.' in name:
            return mean_age_master
        if 'Ms.' in name:
            return mean_age_miss
    else:
        return age

def reasign_cabin(cabin_fare):    
    cabin = cabin_fare[0]
    fare = cabin_fare[1]
    
    if cabin=='X':
        if (fare >= 113.5):
            return 'B'
        if ((fare < 113.5) and (fare > 100)):
            return 'C'
        if ((fare < 100) and (fare > 57)):
            return 'D'
        if ((fare < 57) and (fare > 46)):
            return 'D'
        else:
            return 'X'
    else:
        return cabin

def create_alone_feature(SibSp_Parch):
    if (SibSp_Parch[0]+SibSp_Parch[1])==0:
        return 1
    else:
        return 0