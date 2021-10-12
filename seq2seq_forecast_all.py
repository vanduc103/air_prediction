### ConvLSTM for air pollution forecasting with All spatiotemporal factors

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

def batch_creator(X, X_met, X_tf, X_spd, X_china, batch_size, dataset_length):
    batch_x = list()
    batch_y = list()
    batch_xmet = list()
    batch_xtf = list()
    batch_xsp = list()
    batch_x_china = list()

    """Create batch with random samples and return appropriate format"""
    batch_mask = rng.choice(dataset_length - timesteps - pred_timesteps, batch_size)
    for i in range(len(batch_mask)):
        offset = batch_mask[i]
        
        batch_x.append(X[offset : offset + timesteps])
        batch_xmet.append(X_met[offset : offset + timesteps])
        batch_xtf.append(X_tf[offset : offset + timesteps])
        batch_xsp.append(X_spd[offset : offset + timesteps])

        # create china batch input
        china_input = list()
        for c in range(timesteps):
            # get all data until reaching china_timesteps
            if offset >= china_timesteps:
                china_input.append(X_china[offset - china_timesteps : offset])
            else:
                # pad zero to reach china_timesteps
                x_china = np.zeros((china_timesteps - offset, china_features))
                x_china = np.concatenate([ x_china, X_china[0 : offset] ], axis=0)
                china_input.append(x_china)
        batch_x_china.append(china_input)

        batch_y.append(X[offset + timesteps : offset + timesteps + pred_timesteps])
    
    batch_x = np.asarray(batch_x)
    batch_xmet = np.asarray(batch_xmet)
    batch_xtf = np.asarray(batch_xtf)
    batch_xsp = np.asarray(batch_xsp)
    batch_x_china = np.asarray(batch_x_china)

    batch_x = np.expand_dims(batch_x, axis=3)
    batch_xtf = np.expand_dims(batch_xtf, axis=3)
    batch_xsp = np.expand_dims(batch_xsp, axis=3)
    batch_x = np.concatenate((batch_x, batch_xmet, batch_xtf, batch_xsp), axis=3)

    batch_y = np.asarray(batch_y)
    batch_ymap = np.zeros((batch_size, pred_timesteps, grid_size))
    batch_ymap[:, :, station_map] = 1.0

    batch_x_china = np.transpose(batch_x_china, (0, 2, 1, 3))
    batch_x_china = batch_x_china.reshape((batch_size, china_timesteps, timesteps*china_features))
    batch_x = batch_x.reshape((batch_size, timesteps, image_size, image_size, in_channel))
    batch_y = batch_y.reshape((batch_size, output_size))
    batch_ymap = batch_ymap.reshape((batch_size, output_size))

    return batch_x_china, batch_x, batch_y, batch_ymap

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

# load meteorology data
X_met = np.load('data/meteorology_transformed.npy')
X_met = X_met[0:len(X)]
print(X_met.shape)

# load traffic data
traffic_file = 'data/traffic.h5'
if os.path.isfile(traffic_file):
    with h5py.File(traffic_file, 'r') as hf:
        X_tf = hf['traffic'][:]
X_tf = X_tf[0:len(X)]
print(X_tf.shape)

# normalize features
tf_scaler = MinMaxScaler(feature_range=(0, 1))
X_tf = tf_scaler.fit_transform(X_tf.reshape(X_tf.shape[0]*X_tf.shape[1],1)).reshape(X_tf.shape[0], X_tf.shape[1])

# load speed data
speed_file = 'data/speed.h5'
if os.path.isfile(speed_file):
    with h5py.File(speed_file, 'r') as hf:
        X_spd = hf['speed'][:]
X_spd = X_spd[0:len(X)]
print(X_spd.shape)

# normalize features
spd_scaler = MinMaxScaler(feature_range=(0, 1))
X_spd = spd_scaler.fit_transform(X_spd.reshape(X_spd.shape[0]*X_spd.shape[1],1)).reshape(X_spd.shape[0], X_spd.shape[1])

# load china data
china_file = 'data/china.h5'
if os.path.isfile(china_file):
    with h5py.File(china_file, 'r') as hf:
        X_beijing = hf['beijing'][:]
        X_shanghai = hf['shanghai'][:]
        X_shandong = hf['shandong'][:]

X_beijing = X_beijing[0:len(X)]
X_shanghai = X_shanghai[0:len(X)]
X_shandong = X_shandong[0:len(X)]

# normalize features
scaler_china = MinMaxScaler(feature_range=(0, 1))
X_beijing = scaler_china.fit_transform(X_beijing.reshape(X_beijing.shape[0],1))
X_shanghai = scaler_china.fit_transform(X_shanghai.reshape(X_shanghai.shape[0],1))
X_shandong = scaler_china.fit_transform(X_shandong.reshape(X_shandong.shape[0],1))

X_china = np.concatenate([X_beijing, X_shanghai, X_shandong], axis=1) # 3 features
print(X_china.shape)

# split to train, validate, test set
train_size = (365+366)*24
X_train, X_test = X[:train_size], X[train_size:]
X_met_train, X_met_test = X_met[:train_size], X_met[train_size:]
X_tf_train, X_tf_test = X_tf[:train_size], X_tf[train_size:]
X_spd_train, X_spd_test = X_spd[:train_size], X_spd[train_size:]
X_china_train, X_china_test = X_china[:train_size], X_china[train_size:]

split_size = train_size - (92)*24
X_train, X_val = X_train[:split_size], X_train[split_size:]
X_met_train, X_met_val = X_met_train[:split_size], X_met_train[split_size:]
X_tf_train, X_tf_val = X_tf_train[:split_size], X_tf_train[split_size:]
X_spd_train, X_spd_val = X_spd_train[:split_size], X_spd_train[split_size:]
X_china_train, X_china_val = X_china_train[:split_size], X_china_train[split_size:]

print('Training set shape: {}'.format(X_train.shape))
print('Validate set shape: {}'.format(X_val.shape))
print('Test set shape: {}'.format(X_test.shape))

# Training Parameters
timesteps = 12 # timesteps
pred_timesteps = 12 # predict timesteps
china_timesteps = 100

learning_rate = 0.001
training_steps = 200
batch_size = 128
display_step = 20
is_training = True

# Network Parameters
grid_size = 1024
image_size = 32
in_channel = 1 + X_met.shape[2] + 1 + 1 # pollution + met + traffic + speed
out_channel = [8, 8, 8]
china_features = 3 # 3 cities
n_hidden = 2000 # hidden layer
fc_size = 1000
dr_rate = 0.5
loss_ratio = grid_size/len(station_map)
china_output_size = grid_size * timesteps
output_size = grid_size * pred_timesteps

# tf china input
x_china = tf.placeholder("float", [None, china_timesteps, timesteps*china_features])

# define the model
# Unstack to get a list of 'timesteps' tensors of shape (batch_size, num_features)
input_x = tf.unstack(x_china, china_timesteps, axis=1)

# LSTM layers with n_hidden units.
num_units = [n_hidden]
cells = [rnn.LSTMCell(num_units=n) for n in num_units]
rnn_cell = rnn.MultiRNNCell(cells)

# dropout
#rnn_cell = rnn.DropoutWrapper(rnn_cell, output_keep_prob=dropout)

# generate prediction
outputs, states = rnn.static_rnn(rnn_cell, input_x, dtype=tf.float32)

# fully connected => embedding
W_fc = tf.get_variable(name='W_fc', shape=[n_hidden, fc_size], 
        initializer=tf.contrib.layers.xavier_initializer())
b_fc = tf.Variable(tf.zeros(fc_size))
fc = tf.nn.relu(tf.add(tf.matmul(states[-1][-1], W_fc), b_fc))

# drop out layer
dropout = tf.layers.dropout(
    inputs=fc, rate=dr_rate, training=True)

# output
W_output = tf.get_variable(name='W_out', shape=[fc_size, china_output_size], 
        initializer=tf.contrib.layers.xavier_initializer())
b_output = tf.Variable(tf.zeros(china_output_size))
china_out = tf.nn.sigmoid(tf.add(tf.matmul(dropout, W_output), b_output))

# tf Graph input
x = tf.placeholder("float", [None, timesteps, image_size, image_size, in_channel])
# concat pollution feature with china embedded => #in_channel += 1
china_out = tf.reshape(china_out, [-1, timesteps, image_size, image_size, 1])
x_concat = tf.concat([x, china_out], axis=4)

### define the model
def prediction_model(x):
    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, num_features)
    input_x = tf.unstack(x, timesteps, axis=1)

    # encode convLSTM layer
    conv_cell1 = tf.contrib.rnn.ConvLSTMCell(
                    conv_ndims=2,
                    input_shape=[image_size, image_size, in_channel + 1],
                    output_channels=out_channel[0],
                    kernel_shape=[3, 3],
                    name="conv_lstm_cell1")

    conv_cell2 = tf.contrib.rnn.ConvLSTMCell(
                    conv_ndims=2,
                    input_shape=[image_size, image_size, out_channel[0]],
                    output_channels=out_channel[1],
                    kernel_shape=[3, 3],
                    name="conv_lstm_cell2")

    conv_cell3 = tf.contrib.rnn.ConvLSTMCell(
                    conv_ndims=2,
                    input_shape=[image_size, image_size, out_channel[1]],
                    output_channels=out_channel[2],
                    kernel_shape=[3, 3],
                    name="conv_lstm_cell3")

    conv_cells = rnn.MultiRNNCell([conv_cell1, conv_cell2, conv_cell3])

    # generate prediction
    init_state = conv_cells.zero_state(tf.shape(x)[0], dtype=tf.float32)
    outputs, states = rnn.static_rnn(conv_cells, input_x, initial_state=init_state, dtype=tf.float32)

    # decode convLSTM layer
    deconv_cell1 = tf.contrib.rnn.ConvLSTMCell(
                    conv_ndims=2,
                    input_shape=[image_size, image_size, out_channel[2]],
                    output_channels=out_channel[0],
                    kernel_shape=[3, 3],
                    name="deconv_lstm_cell1")

    deconv_cell2 = tf.contrib.rnn.ConvLSTMCell(
                    conv_ndims=2,
                    input_shape=[image_size, image_size, out_channel[0]],
                    output_channels=out_channel[1],
                    kernel_shape=[3, 3],
                    name="deconv_lstm_cell2")

    deconv_cell3 = tf.contrib.rnn.ConvLSTMCell(
                    conv_ndims=2,
                    input_shape=[image_size, image_size, out_channel[1]],
                    output_channels=out_channel[2],
                    kernel_shape=[3, 3],
                    name="deconv_lstm_cell3")

    deconv_cells = rnn.MultiRNNCell([deconv_cell1, deconv_cell2, deconv_cell3])

    # generate prediction
    init_state = states
    input_deconv = tf.unstack(tf.transpose(tf.reshape(outputs, [-1, tf.shape(outputs)[1], image_size, image_size, out_channel[2]]), [1, 0, 2, 3, 4]), timesteps, axis=1)
    outputs, states = rnn.static_rnn(deconv_cells, input_deconv, initial_state=init_state, dtype=tf.float32)

    # fully connected
    conv_input = tf.reshape(states[-1][-1], [-1, image_size, image_size, out_channel[2]])
    flatten = tf.contrib.layers.flatten(conv_input)
    flatten_dim = flatten.get_shape()[1].value
    W_fc = tf.get_variable(name='W_fc', shape=[flatten_dim, fc_size], 
		    initializer=tf.contrib.layers.xavier_initializer())
    b_fc = tf.Variable(tf.zeros(fc_size))
    fc = tf.nn.relu(tf.add(tf.matmul(flatten, W_fc), b_fc))

    # drop out layer
    dropout = tf.layers.dropout(
        inputs=fc, rate=dr_rate, training=True)

    # output layer
    W_output = tf.get_variable(name='W_output', shape=[fc_size, output_size],   
                    initializer=tf.contrib.layers.xavier_initializer())
    b_output = tf.Variable(tf.zeros(output_size))
    output = tf.add(tf.matmul(dropout, W_output), b_output)

    # l2 regularization
    l2 = tf.nn.l2_loss(W_output)

    return output, l2

with tf.variable_scope('prediction_model'):
    output, out_l2 = prediction_model(x_concat)

# tf Graph output
y = tf.placeholder("float", [None, output_size])
y_map = tf.placeholder("float", [None, output_size])

# Loss and optimizer
if is_training:
    yhat = tf.multiply(output, y_map)
    loss = tf.sqrt(tf.losses.mean_squared_error(labels=y, predictions=yhat) * loss_ratio)

    beta = 0.01
    regularizer = out_l2
    loss = tf.reduce_mean(loss + beta * regularizer)
    tf.summary.scalar('seq2seq_forecast_all_loss', loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Initializing the variables
init = tf.global_variables_initializer()

# 'Saver' op to save and restore all the variables
saverEmbedd = tf.train.Saver([W_fc, b_fc, W_output, b_output])
saver = tf.train.Saver()
model_china = "model/forecast_china_embedding.ckpt"
model_path = "model/seq2seq_forecast_all.ckpt"

# Restore only the layers up to fc1 (included)
# Calling function `init_fn(sess)` will load all the pretrained weights.
variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=None)
init_fn = tf.contrib.framework.assign_from_checkpoint_fn(model_china, variables_to_restore)

# Merge all the summaries and write them out
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('log/seq2seq_forecast_all_train', flush_secs=10)
val_writer = tf.summary.FileWriter('log/seq2seq_forecast_all_val', flush_secs=10)

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

    if is_training:
        saverEmbedd.restore(sess, model_china)

        for step in range(1, training_steps+1):
            # Make the training batch for each step
            batch_x_china, batch_x, batch_y, batch_ymap = batch_creator(X_train, X_met_train, X_tf_train, X_spd_train, X_china_train, batch_size, X_train.shape[0])

            # Run optimization
            _, train_loss, summary = sess.run([optimizer, loss, merged], feed_dict = {x_china: batch_x_china, x: batch_x, y: batch_y, y_map: batch_ymap}, options=run_options)
            #train_writer.add_summary(summary, step)

            # compute error on validate set
            batch_x_china, batch_x, batch_y, batch_ymap = batch_creator(X_val, X_met_val, X_tf_val, X_spd_val, X_china_val, batch_size, X_val.shape[0])

            [validate_loss, summary] = sess.run([loss, merged], feed_dict={x_china: batch_x_china, x: batch_x, y: batch_y, y_map: batch_ymap}, options=run_options)
            #val_writer.add_summary(summary, step)

            # Print result
            if step % display_step == 0 or step == 1:
                print("Iter = " + str(step) + ". Train error = {:.6f}. Validate error = {:.6f}".format(train_loss, validate_loss))
                    
        print("Training Finished!")
        # Save model weights to disk
        save_path = saver.save(sess, model_path)
        print("Model saved in file: %s" % save_path)

    # Test error
    saver.restore(sess, model_path)
    loss_test = 0
    elapsed_time = 0
    batch_size = 1
    test_steps = int(X_test.shape[0] / batch_size)
    for i in range(test_steps):
        batch_x_china, batch_x, batch_y, batch_ymap = batch_creator(X_test, X_met_test, X_tf_test, X_spd_test, X_china_test, batch_size, X_test.shape[0])
        
        start_time = time.time()
        [out] = sess.run([output], feed_dict={x_china: batch_x_china, x: batch_x, y: batch_y, y_map: batch_ymap})

        inv_out = scaler.inverse_transform(out.flatten().reshape(-1, 1))
        inv_yhat = np.multiply(inv_out, batch_ymap.flatten().reshape(-1, 1))
        inv_y = scaler.inverse_transform(batch_y.flatten().reshape(-1, 1))
        loss_value = sqrt(mean_squared_error(inv_y, inv_yhat) * loss_ratio)

        if i % 600 == 0:
            print(loss_value)
            '''fig = plt.figure()
            ax = fig.add_subplot(121)
            ax.set_title('prediction')
            X_i = inv_out.reshape(32, 32)
            ax.imshow(X_i, cmap='gray', interpolation='none')

            ax = fig.add_subplot(122)
            ax.set_title('actual')
            X_i1 = inv_y.reshape(32, 32)
            ax.imshow(X_i1, cmap='gray', interpolation='none')'''

            #plt.show()

        elapsed_time += time.time() - start_time
        loss_test += loss_value

    # Print test error
    print("Test Error = {:.6f}. Elapsed time = {:.3f}".format(loss_test/test_steps, elapsed_time/test_steps))

    output_path = "output/seq2seq_forecast_all.txt"
    outfile = open(output_path, 'a')
    outfile.write('\n')
    outfile.write("Result with timesteps = " + str(timesteps) + ", predict timesteps = " + str(pred_timesteps) + ", out_channel = " + str(out_channel) + "\n")
    outfile.write("Test Error = {:.6f}. Elapsed time = {:.3f}\n".format(loss_test/test_steps, elapsed_time/test_steps))

