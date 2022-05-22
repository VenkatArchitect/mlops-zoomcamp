from pickletools import int4
import pandas as pd
import datetime
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

#read dataframe function
def read_dataframe(filename):
    df = pd.read_parquet(filename)

    print("Answer 1: No. of records: ", len(df), " in file ", filename)

    df['duration'] = df['dropOff_datetime'] - df['pickup_datetime']

    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    print("Answer 1 of Question 2: Average ride time: ", df.duration.sum() / len(df))

    df2 = df[(df.duration >= 1.0) & (df.duration <= 60.0)]

    print("Answer 2 of Question 2: No. of records dropped: ", len(df) - len(df2))

    df2['PUlocationID'] = df2['PUlocationID'].fillna(-1)
    df2['DOlocationID'] = df2['DOlocationID'].fillna(-1)

    categorical = ['PUlocationID', 'DOlocationID']
    df2[categorical] = df2[categorical].astype(int) #convert to int first to avoid "-1.0" as the value (float)
    df2[categorical] = df2[categorical].astype(str)

    total = len(df2)

    df2.PUlocID_missing_values = ((total - (total - (df2['PUlocationID'] == '-1').sum())) / total) * 100
    df2.DOlocID_missing_values = ((total - (total - (df2['DOlocationID'] == '-1').sum())) / total) * 100

    print("Answer 1 of Question 3: No. of missing values in pickup location ID: ", df2.PUlocID_missing_values)
    print("Answer 2 of Question 3: No. of missing values in dropoff location ID: ", df2.DOlocID_missing_values)

    return df2


#Start of main code

df_train = read_dataframe("C:\\Users\\vrama\\Documents\\Projects\\Files\\fhv_tripdata_2021-01.parquet")
df_val = read_dataframe("C:\\Users\\vrama\\Documents\\Projects\\Files\\fhv_tripdata_2021-02.parquet")

factors = ['PUlocationID', 'DOlocationID']
train_dicts = df_train[factors].to_dict(orient = 'records')
dv = DictVectorizer ()
X_train = dv.fit_transform(train_dicts)

print("Answer 4.1: Dimensionality of matrix in training data:  ", len(dv.feature_names_))

val_dicts = df_val[factors].to_dict(orient = 'records')
X_val = dv.transform(val_dicts)

print("Answer 4.2: Dimensionality of matrix in validation data:  ", len(dv.feature_names_))

target = 'duration'
Y_train = df_train[target].values
Y_val = df_val[target].values

lr = LinearRegression()
lr.fit (X_train, Y_train)

Y_pred = lr.predict(X_train)

print("Answer 5: Mean squared error on prediction of training (jan 2021) data: ", mean_squared_error (Y_train, Y_pred, squared=False))

Y_pred_feb = lr.predict(X_val)

print("Answer 6: Mean Squared Error on prediction of validation (feb 2021) data: ", mean_squared_error (Y_val, Y_pred_feb, squared=False))
