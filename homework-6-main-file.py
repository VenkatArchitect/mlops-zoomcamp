#!/usr/bin/env python
# coding: utf-8

import os
import sys
import pickle
import pandas as pd

def get_input_path(year, month):
    default_input_pattern = 's3://nyc-duration/in/fhv_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)


def get_output_path(year, month):
    default_output_pattern = 's3://nyc-duration/out/{year:04d}-{month:02d}.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)


def read_data(inp_filename, categorical):

    S3_ENDPOINT_URL = 'http://localhost:4566'

    options = {
            'client_kwargs': {
                'endpoint_url': S3_ENDPOINT_URL
            }
    }

    print("Inside read_data")
    print(inp_filename)
    print(options)
    
    
    df = pd.read_parquet(inp_filename, storage_options=options)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def main(year, month):

    year = int(year)
    month = int(month)

    print(year)
    print(month)

    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)

    print(input_file)
    print(output_file)

    #with open('model.bin', 'rb') as f_in:
    #    dv, lr = pickle.load(f_in)

    categorical = ['PUlocationID', 'DOlocationID']

    df = read_data(input_file, categorical)

    ### NOTE: Getting an exception in read_data above and could not proceed further.
    # The exception complains about unable to connect to the endpoint URL in spite of
    # endpoint URL being correct and accessible (localstack s3)
    #
    # I am attaching the docker-compose.yaml file also for proof

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print('predicted mean duration:', y_pred.mean())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    df_result.to_parquet(output_file, engine='pyarrow', index=False)

#Q1 answer :
if __name__ == "__main__":
    main("2021","02")
