
import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 800)
pd.set_option('display.max_columns', 500)

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set()

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report 
from sklearn.metrics import accuracy_score 
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import RidgeClassifier

from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_curve

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

import re

df = pd.read_csv("C:/Users/lenovo/Desktop/ML Project DataKlub 2/music_genre.csv")

df = df.drop_duplicates()

df.dropna(inplace=True)

df.drop(["instance_id", "obtained_date", "track_name", "duration_ms"], axis=1, inplace=True)

df = df[df['artist_name'] != 'empty_field']

df = df.drop(df[df["tempo"] == "?"].index)

df["tempo"] = df["tempo"].astype("float64")


df = df.reset_index(drop=True)

# create a dictionary of replacements
replacements = {'Minor': 0, 'Major': 1}

# replace values using the .map() method
df['mode'] = df['mode'].map(replacements)

ohe = OneHotEncoder(handle_unknown = 'ignore', sparse_output=False).set_output(transform = "pandas")

ohetransform = ohe.fit_transform(df[['key']])

df = pd.concat([df, ohetransform], axis=1).drop(columns=['key'])


# split data set to 60/20/20 for validation and testing
data_full_train, data_test = train_test_split(df, test_size=0.2, random_state=42)
data_train, data_val = train_test_split(data_full_train, test_size=0.25, random_state=42)


y_train = data_train.music_genre.values
y_val = data_val.music_genre.values
y_test = data_test.music_genre.values

del data_train['music_genre']
del data_val['music_genre']
del data_test['music_genre']


dv = DictVectorizer(sparse=False)

train_dict = data_train.to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = data_val.to_dict(orient='records')
X_val = dv.transform(val_dict)


dicts_full_train = data_full_train.to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dicts_full_train)

y_full_train = data_full_train.music_genre.values

modelrcl = RidgeClassifier(alpha=1, random_state=42)
modelrcl.fit(X_full_train, y_full_train)

dicts_test = data_test.to_dict(orient='records')
X_test = dv.transform(dicts_test)

lr_pred = modelrcl.predict(X_test)
lr_class = classification_report(y_test, lr_pred)
lr_score = accuracy_score(y_test, lr_pred)
print(lr_class)
print(f'\n Accuracy score of {np.round(lr_score, 2)} with Ridge Classifier')

import pickle


# In[360]:


with open('modelrcl.pkl', 'wb') as f_out:
    pickle.dump((dv, modelrcl), f_out)

print(f'the model is saved to modelrcl.pkl')

