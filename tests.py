from libraries import apply_transformations

from datetime import datetime
import pandas as pd

def dt(hour, minute, second=0):
    return datetime(2021, 1, 1, hour, minute, second)

input_data = [
    (None, None, dt(1, 2), dt(1, 10)),
    (1, 1, dt(1, 2), dt(1, 10)),
    (1, 1, dt(1, 2, 0), dt(1, 2, 50)),
    (1, 1, dt(1, 2, 0), dt(2, 2, 1)),        
]

#Q1: Answer is the script 'homework-6-main-file.py' where the main function has been re-organized.

#Q2: Answer: __init__.py

df = pd.DataFrame(input_data, columns = ['PUlocationID', 'DOlocationID', 'pickup_datetime', 'dropOff_datetime'])

categorical = ['PUlocationID', 'DOlocationID']

df1 = apply_transformations(df, categorical)

#actual_data = df.to_dict('records')

df2 =   pd.DataFrame({'PUlocationID' : ['-1','1','1'], 
         'DOlocationID' : ['-1','1','1'],
         'pickup_datetime' : [pd.to_datetime('2021-01-01 01:02:00'), pd.to_datetime('2021-01-01 01:02:00'), pd.to_datetime('2021-01-01 01:02:00')],
         'dropOff_datetime' : [pd.to_datetime('2021-01-01 01:10:00'), pd.to_datetime('2021-01-01 01:10:00'), pd.to_datetime('2021-01-01 01:02:50')],
         'duration' : [float('8.000000'), float('8.000000'), float('0.833333')]})

pd.testing.assert_frame_equal(df1, df2) #This assert function gives output only if there's a difference.

#Q3 : Answer : 3 rows











