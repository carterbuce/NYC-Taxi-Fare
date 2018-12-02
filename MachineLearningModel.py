# Python 3
import numpy as np  # linear algebra
import pandas as pd  # CSV file I/O (e.g. pd.read_csv)
import os  # reading the input files we have access to
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Sequential, optimizers, regularizers, backend
from sklearn import preprocessing
from tensorflow.python.keras.layers import BatchNormalization, Dense
from Util import *

# Model hyperparameters
BATCH_SIZE = 350  # larger leads to generalization, smaller leads to overfit model
EPOCHS = 12  # how many times to run all the data through the model
LEARNING_RATE = 0.001

train_df = pd.read_csv('input/train_clean.csv')
test_df = pd.read_csv('input/test.csv')


# add more features
add_trip_dist(test_df)
add_trip_dist(train_df)
add_dist_airports(test_df)
add_dist_airports(train_df)
add_dist_manhattan(test_df)
add_dist_manhattan(train_df)

features = ['abs_diff_distance',
            'dist_pickup_jfk', 'dist_dropoff_jfk',
            'dist_pickup_lga', 'dist_dropoff_lga',
            'dist_pickup_ewr', 'dist_dropoff_ewr',
            'dist_pickup_manhattan', 'dist_dropoff_manhattan'
            ]

train_features = train_df[features]
test_features = test_df[features]
train_labels = train_df['fare_amount'].values


# Scale data so each feature is between 0 and 1
scaler = preprocessing.MinMaxScaler()
train_scaled = scaler.fit_transform(train_features)
test_scaled = scaler.transform(test_features)

# create layers of our neural network
model = Sequential()
model.add(Dense(256, activation='relu', input_dim=train_scaled.shape[1], activity_regularizer=regularizers.l1(0.01)))
model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(8, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1))

adam = optimizers.adam(lr=LEARNING_RATE)
model.compile(loss='mse', optimizer=adam, metrics=['mae'])  # mae == rmse

history = model.fit(x=train_scaled, y=train_labels,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    verbose=2, validation_split=0.10,
                    shuffle=True)

prediction = model.predict(test_scaled, batch_size=64, verbose=1)  # batch size here doesn't matter

# save our predictions to submission.csv
save_submission(test_df, prediction, 'key', 'fare_amount', 'submission.csv')
