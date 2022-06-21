import pickle
import statistics
import pandas as pd
import sys


def read_data(filename, categorical):
    df = pd.read_parquet(filename)

   # the below two lines are required while reading from csv
   # df.dropOff_datetime = pd.to_datetime(df.dropOff_datetime)
   # df.pickup_datetime = pd.to_datetime(df.pickup_datetime)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df



def main ():
    args = sys.argv[1:]
    year = args[0]
    month = args[1]

    print(year, month)
    
    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    categorical = ['PUlocationID', 'DOlocationID']

    filename_inp = 'fhv_tripdata_' + year + '-' + month + '.parquet'
    df = read_data (filename_inp, categorical)
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print(statistics.mean(y_pred))

if __name__ == "__main__":
    main()





#print(y_pred)
