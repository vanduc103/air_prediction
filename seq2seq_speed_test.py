### Compute spRMSE for air pollution interpolation with driving speed spatiotemporal factor

from __future__ import print_function

from read_data import read_data

import tensorflow as tf
from tensorflow.contrib import rnn

import numpy as np
import pandas as pd
import h5py

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt

import random
import time
import os

# random number
seed = 128
rng = np.random.RandomState(seed)

def batch_creator(X, X_spd, batch_size, dataset_length, s):
    batch_x = list()
    batch_y = list()
    batch_xspd = list()

    """Create batch with random samples and return appropriate format"""
    batch_mask = rng.choice(dataset_length - timesteps - pred_timesteps, batch_size)
    for i in range(len(batch_mask)):
        offset = batch_mask[i]
        
        batch_x.append(X[offset : offset + timesteps])
        batch_xspd.append(X_spd[offset : offset + timesteps])
        batch_y.append(X[offset + timesteps : offset + timesteps + pred_timesteps])
    
    batch_x = np.asarray(batch_x)
    batch_x[:, :, station_map[s]] = 0
    batch_xspd = np.asarray(batch_xspd)
    batch_x = np.expand_dims(batch_x, axis=3)
    batch_xspd = np.expand_dims(batch_xspd, axis=3)
    batch_x = np.concatenate((batch_x, batch_xspd), axis=3)

    batch_y = np.asarray(batch_y)
    batch_ymap = np.zeros((batch_size, output_size))
    batch_ymap[:, station_map] = 1.0

    batch_x = batch_x.reshape((batch_size, timesteps, image_size, image_size, in_channel))
    batch_y = batch_y.reshape((batch_size, output_size))

    return batch_x, batch_y, batch_ymap

# load pollution data
pollution_file = 'data/pollutionPM25.h5'
if os.path.isfile(pollution_file):
    with h5py.File(pollution_file, 'r') as hf:
        X = hf['pollution'][:]
        station_map = hf['station_map'][:]

print(X.shape)

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X.reshape(X.shape[0]*X.shape[1],1)).reshape(X.shape[0], X.shape[1])

# load traffic data
speed_file = 'data/speed.h5'
if os.path.isfile(speed_file):
    with h5py.File(speed_file, 'r') as hf:
        X_spd = hf['speed'][:]
X_spd = X_spd[0:len(X)]
print(X_spd.shape)

# normalize features
spd_scaler = MinMaxScaler(feature_range=(0, 1))
X_spd = spd_scaler.fit_transform(X_spd.reshape(X_spd.shape[0]*X_spd.shape[1],1)).reshape(X_spd.shape[0], X_spd.shape[1])

# split to train, validate, test set
train_size = (365+366)*24
X_train, X_test = X[:train_size], X[train_size:]
X_spd_train, X_spd_test = X_spd[:train_size], X_spd[train_size:]
split_size = train_size - (92)*24
X_train, X_val = X_train[:split_size], X_train[split_size:]
X_spd_train, X_spd_val = X_spd_train[:split_size], X_spd_train[split_size:]
print('Training set shape: {}'.format(X_train.shape))
print('Validate set shape: {}'.format(X_val.shape))
print('Test set shape: {}'.format(X_test.shape))

# Training Parameters
timesteps = 1 # timesteps
pred_timesteps = 1 # predict timesteps
learning_rate = 0.001
training_steps = 200
batch_size = 128
display_step = 20
is_training = True

# Network Parameters
grid_size = 1024
image_size = 32
in_channel = 1 + 1 # pollution + speed
out_channel = [64]
n_hidden = 1000 # hidden layer
fc_size = 1000
dr_rate = 0.5
loss_ratio = grid_size/len(station_map)
output_size = grid_size

# tf Graph input
x = tf.placeholder("float", [None, timesteps, image_size, image_size, in_channel])

### define the model
def prediction_model(x):
    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, num_features)
    input_x = tf.unstack(x, timesteps, axis=1)

    # encode convLSTM layer
    conv_cell = tf.contrib.rnn.ConvLSTMCell(
                    conv_ndims=2,
                    input_shape=[image_size, image_size, in_channel],
                    output_channels=out_channel[0],
                    kernel_shape=[3, 3],
                    name="conv_lstm_cell")

    # generate prediction
    init_state = conv_cell.zero_state(tf.shape(x)[0], dtype=tf.float32)
    outputs, states = rnn.static_rnn(conv_cell, input_x, initial_state=init_state, dtype=tf.float32)

    # decode convLSTM layer 1
    deconv_cell = tf.contrib.rnn.ConvLSTMCell(
                    conv_ndims=2,
                    input_shape=[image_size, image_size, out_channel[0]],
                    output_channels=out_channel[0],
                    kernel_shape=[3, 3],
                    name="deconv_lstm_cell")

    # generate prediction
    init_state = states
    input_deconv = tf.unstack(tf.transpose(tf.reshape(outputs, [-1, tf.shape(outputs)[1], image_size, image_size, out_channel[0]]), [1, 0, 2, 3, 4]), timesteps, axis=1)
    outputs, states = rnn.static_rnn(deconv_cell, input_deconv, initial_state=init_state, dtype=tf.float32)

    # 1x1 convolutional
    conv_input = tf.reshape(states[-1], [-1, image_size, image_size, out_channel[0]])
    W_output = tf.get_variable(name='W_output', shape=[1, 1, out_channel[0], pred_timesteps],   
                initializer=tf.contrib.layers.xavier_initializer())
    b_output = tf.Variable(tf.zeros(pred_timesteps))
    output = tf.nn.sigmoid(tf.nn.conv2d(conv_input, W_output, strides=[1,1,1,1], padding='SAME') + b_output)
    output = tf.reshape(output, [-1, output_size])

    # l2 regularization
    l2 = tf.nn.l2_loss(W_output)

    return output, l2

with tf.variable_scope('prediction_model'):
    output, out_l2 = prediction_model(x)

# tf Graph output
y = tf.placeholder("float", [None, output_size])
y_map = tf.placeholder("float", [None, output_size])

# Initializing the variables
init = tf.global_variables_initializer()

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()
model_path = "model/seq2seq_spd.ckpt"

# Merge all the summaries and write them out
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('log/seq2seq_spd_train', flush_secs=10)
val_writer = tf.summary.FileWriter('log/seq2seq_spd_val', flush_secs=10)

print("Result with timesteps = " + str(timesteps) + ", predict timesteps = " + str(pred_timesteps) + ", out_channel = " + str(out_channel) + "\n")

# Soft placement allows placing on CPU ops without GPU implementation.
session_config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False)
session_config.gpu_options.per_process_gpu_memory_fraction = 0.9
session_config.gpu_options.visible_device_list = '2,3'
run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)

# Start training
with tf.Session(config=session_config) as sess:
    # Run the initializer
    sess.run(init)

    # Restore trained model
    saver.restore(sess, model_path)

    # Test error
    total_loss_test = 0
    total_elapsed_time = 0
    batch_size = 128
    test_steps = int(X_test.shape[0] / batch_size)

    for s in range(len(station_map)):
        print(station_map[s])
        loss_test = 0
        elapsed_time = 0
        for i in range(test_steps):
            batch_x, batch_y, batch_ymap = batch_creator(X_test, X_spd_test, batch_size, X_test.shape[0], s)
            
            start_time = time.time()
            [out_test] = sess.run([output], feed_dict={x: batch_x, y: batch_y, y_map: batch_ymap})

            inv_out = scaler.inverse_transform(out_test.flatten().reshape(-1, 1))
            inv_yhat = inv_out[station_map[s], :]
            inv_y = scaler.inverse_transform(batch_y.flatten().reshape(-1, 1))
            inv_y = inv_y[station_map[s], :]
            loss_value = sqrt(mean_squared_error(inv_y, inv_yhat))

            elapsed_time += time.time() - start_time
            loss_test += loss_value
        # Print test error
        print("Test Error = {:.6f}. Elapsed time = {:.3f}".format(loss_test/test_steps, elapsed_time/test_steps))
        total_loss_test += loss_test
        total_elapsed_time += elapsed_time

    # Print validate error
    test_steps = test_steps * len(station_map)
    print("spRMSE = {:.6f}. Elapsed time = {:.3f}".format(total_loss_test/test_steps, total_elapsed_time/test_steps))

