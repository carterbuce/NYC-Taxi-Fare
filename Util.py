import numpy as np
import pandas as pd

# Given a dataframe, add two new features 'abs_diff_longitude' and
# 'abs_diff_latitude' reprensenting the "Manhattan vector" from
# the pickup location to the dropoff location.
def add_trip_dist(df):
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()
    df['abs_diff_distance'] = df.abs_diff_latitude + df.abs_diff_longitude  # manhattan distance
    # df['abs_diff_distance'] = np.sqrt(np.power(df.abs_diff_longitude, 2) + np.power(df.abs_diff_latitude, 2))


def add_dist_jfk(df):
    lat_jfk = 40.6413
    long_jfk = -73.7781

    df['dist_pickup_jfk'] = abs(df.pickup_longitude - long_jfk) + abs(df.pickup_latitude - lat_jfk)
    df['dist_dropoff_jfk'] = abs(df.dropoff_longitude - long_jfk) + abs(df.dropoff_latitude - lat_jfk)

def add_dist_lga(df):
    lat_lga = 40.7769
    long_lga = -73.8740

    df['dist_pickup_lga'] = abs(df.pickup_longitude - long_lga) + abs(df.pickup_latitude - lat_lga)
    df['dist_dropoff_lga'] = abs(df.dropoff_longitude - long_lga) + abs(df.dropoff_latitude - lat_lga)

def add_dist_ewr(df):
    lat_ewr = 40.6895
    long_ewr = -74.1745

    df['dist_pickup_ewr'] = abs(df.pickup_longitude - long_ewr) + abs(df.pickup_latitude - lat_ewr)
    df['dist_dropoff_ewr'] = abs(df.dropoff_longitude - long_ewr) + abs(df.dropoff_latitude - lat_ewr)


def add_dist_airports(df):
    add_dist_jfk(df)
    add_dist_lga(df)
    add_dist_ewr(df)

def add_dist_manhattan(df):
    lat_manhattan = 40.7831  # 40.7458
    long_manhattan = -73.9712  #73.9899
    
    df['dist_pickup_manhattan'] = abs(df.pickup_longitude - long_manhattan) + abs(df.pickup_latitude - lat_manhattan)
    df['dist_dropoff_manhattan'] = abs(df.dropoff_longitude - long_manhattan) + abs(df.dropoff_latitude - lat_manhattan)


def save_submission(raw_test, prediction, id_column, prediction_column, file_name):
    df = pd.DataFrame(prediction, columns=[prediction_column])
    df[id_column] = raw_test[id_column]
    df[[id_column, prediction_column]].to_csv((file_name), index=False)
    print('submission.csv saved')


