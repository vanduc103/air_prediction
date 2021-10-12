### Deep Air Learning reimplemented for Seoul data

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
from math import sqrt, exp

import random
import time
import os

# random number
seed = 128
rng = np.random.RandomState(seed)

def batch_creator_ae(X, batch_size, dataset_length):
    batch_x = list()
    batch_y = list()

    """Create batch with random samples and return appropriate format"""
    batch_mask = rng.choice(dataset_length - timesteps - pred_timesteps, batch_size)
    for i in range(len(batch_mask)):
        offset = batch_mask[i]
        
        batch_x.append(X[offset : offset + timesteps])
        batch_y.append(X[offset : offset + timesteps])
    
    batch_x = np.asarray(batch_x)
    batch_y = np.asarray(batch_y)

    batch_x = batch_x.reshape((batch_size, input_ae))
    batch_y = batch_y.reshape((batch_size, output_ae))

    return batch_x, batch_y

def batch_creator_all(X, batch_size, dataset_length):
    batch_x = list()
    batch_y = list()

    """Create batch with random samples and return appropriate format"""
    batch_mask = rng.choice(dataset_length - timesteps - pred_timesteps, batch_size)
    for i in range(len(batch_mask)):
        offset = batch_mask[i]
        
        batch_x.append(X[offset : offset + timesteps])
        batch_y.append(X[offset + timesteps : offset + timesteps + pred_timesteps])
    
    batch_x = np.asarray(batch_x)
    batch_y = np.asarray(batch_y)
    batch_ymap = np.zeros((batch_size, grid_size))
    batch_ymap[:, station_map] = 1.0

    batch_x = batch_x.reshape((batch_size, input_ae))
    batch_y = batch_y.reshape((batch_size, grid_size))

    return batch_x, batch_y, batch_ymap

# find neighbors of an element
def find_neighbor(e):
    i = int(e / 32)
    j = e - i*32
    # find neighbor of e (max 8 neighbors)
    if i == 0:
        i_nei = [0, 1]
    elif i == 31:
        i_nei = [i-1, i]
    else:
        i_nei = [i-1, i, i+1]
    if j == 0:
        j_nei = [0, 1]
    elif j == 31:
        j_nei = [j-1, j]
    else:
        j_nei = [j-1, j, j+1]
    e_nei = list()
    for t in range(len(i_nei)):
        for k in range(len(j_nei)):
            nei_idx = i_nei[t] * 32 + j_nei[k]
            if nei_idx != e:
                e_nei.append(nei_idx)
    return e_nei

# create a dict of neighbors of all elements
def neighbors():
    my_dict = {}
    for i in range(grid_size):
       nei = find_neighbor(i)
       my_dict[i] = nei
    return my_dict

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

# split to train, validate, test set
train_size = (365+366)*24
X_train, X_test = X[:train_size], X[train_size:]
split_size = train_size - (92)*24
X_train, X_val = X_train[:split_size], X_train[split_size:]
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
is_training_ae = False
is_training_all = False

# Network Parameters
grid_size = 1024
# autoencoder layer
input_ae = grid_size * timesteps
output_ae = input_ae
n_autoencoder = 2000
ae_layers = 4
# spatiotemporal semi-supervised regression
spa = 2
tem = 2
alpha = 2
beta = 3
loss_ratio = grid_size / len(station_map)
output_size = grid_size * (1 + tem)

# tf Graph input
x = tf.placeholder("float", [None, input_ae])

# Autoencoder model
W_ae0 = tf.get_variable(name='W_ae0', shape=[input_ae, n_autoencoder], 
        initializer=tf.contrib.layers.xavier_initializer())
b_ae0 = tf.Variable(tf.zeros(n_autoencoder))
ae0 = tf.nn.relu(tf.add(tf.matmul(x, W_ae0), b_ae0))

W_ae1 = tf.get_variable(name='W_ae1', shape=[n_autoencoder, n_autoencoder], 
        initializer=tf.contrib.layers.xavier_initializer())
b_ae1 = tf.Variable(tf.zeros(n_autoencoder))
ae1 = tf.nn.relu(tf.add(tf.matmul(ae0, W_ae1), b_ae1))

W_ae2 = tf.get_variable(name='W_ae2', shape=[n_autoencoder, n_autoencoder], 
        initializer=tf.contrib.layers.xavier_initializer())
b_ae2 = tf.Variable(tf.zeros(n_autoencoder))
ae2 = tf.nn.relu(tf.add(tf.matmul(ae1, W_ae2), b_ae2))

W_ae3 = tf.get_variable(name='W_ae3', shape=[n_autoencoder, n_autoencoder], 
        initializer=tf.contrib.layers.xavier_initializer())
b_ae3 = tf.Variable(tf.zeros(n_autoencoder))
ae_embed = tf.nn.relu(tf.add(tf.matmul(ae2, W_ae3), b_ae3))

# output layer
W_output = tf.get_variable(name='W_output', shape=[n_autoencoder, output_ae],   
                initializer=tf.contrib.layers.xavier_initializer())
b_output = tf.Variable(tf.zeros(output_ae))
# use sigmoid function to make output in (0,1) but not 0
ae_out = tf.nn.sigmoid(tf.add(tf.matmul(ae_embed, W_output), b_output))

# regularization
ae_l2 = tf.nn.l2_loss(W_ae0) + tf.nn.l2_loss(W_ae1) + tf.nn.l2_loss(W_ae2) + tf.nn.l2_loss(W_ae3)

def stsr_model(x):
    # neural network
    W_stsr = tf.get_variable(name='W_stsr', shape=[n_autoencoder, output_size], 
                initializer=tf.contrib.layers.xavier_initializer())
    b_stsr = tf.Variable(tf.zeros(output_size))
    output = tf.nn.sigmoid(tf.add(tf.matmul(x, W_stsr), b_stsr))

    # regularization
    l2 = tf.nn.l2_loss(W_stsr)

    return output, l2

with tf.variable_scope('stsr_model'):
    output, stsr_l2 = stsr_model(ae_embed)

# tf Graph output
y_ae = tf.placeholder("float", [None, output_ae])

if is_training_ae:
    # Loss and optimizer
    loss_ae = tf.sqrt(tf.losses.mean_squared_error(labels=y_ae, predictions=ae_out))
    reg_beta = 0.01
    regularizer = ae_l2
    loss_ae = tf.reduce_mean(loss_ae + reg_beta * regularizer)
    tf.summary.scalar('dal_loss', loss_ae)

    optimizer_ae = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_ae)

# tf Graph output
y = tf.placeholder("float", [None, grid_size])
y_map = tf.placeholder("float", [None, grid_size])

# reshape output to shape: batch, grid_size, interpolation+temporal
output_re = tf.reshape(output, [-1, grid_size, 1+tem])
output_spa = output_re[:, :, tem:tem+1]
output_spa = tf.reshape(output_spa, [-1, grid_size])

if is_training_all:
    # loss of labelled training data
    yhat = tf.multiply(output_spa, y_map)
    loss1 = tf.sqrt(tf.losses.mean_squared_error(labels=y, predictions=yhat) * loss_ratio)

    # Spatial loss
    loss_spa = 0.0
    labels = tf.slice(output_spa, [0, 0], [-1, 1])
    predictions = tf.slice(output_spa, [0, 1], [-1, 1])
    all_nei = neighbors()
    for e in range(grid_size):
        e_nei = all_nei[e]
        for i in range(spa):
            if e == 0 and i == 0:
                continue
            labels = tf.concat([ labels, tf.slice(output_spa, [0, e], [-1, 1]) ], axis=1)
            predictions = tf.concat([ predictions, tf.slice(output_spa, [0, e_nei[i]], [-1, 1]) ], axis=1)
        
    loss_spa = tf.sqrt(tf.losses.mean_squared_error(labels=labels, predictions=predictions)) * exp(-spa)

    # Temporal loss
    loss_tem = 0.0
    output_tem = output_re[:, :, 0:tem]
    labels = tf.slice(output_spa, begin=[0, 0], size=[-1, 1])
    predictions = tf.slice(output_tem[:,:,0], begin=[0, 0], size=[-1, 1])
    for e in range(grid_size):
        for i in range(tem):
            if e == 0 and i == 0:
                continue
            labels = tf.concat([ labels, tf.slice(output_spa, [0, e], [-1, 1]) ], axis=1)
            predictions = tf.concat([ predictions, tf.slice(output_tem[:,:,i], [0, e], [-1, 1]) ], axis=1)
        
    loss_tem = tf.sqrt(tf.losses.mean_squared_error(labels=labels, predictions=predictions)) * exp(-tem)
    
    # final loss
    loss = loss1 + alpha*loss_spa + beta*loss_tem

    reg_beta = 0.01
    regularizer = stsr_l2
    loss = tf.reduce_mean(loss + reg_beta * regularizer)
    tf.summary.scalar('dal_loss', loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Initializing the variables
init = tf.global_variables_initializer()

# 'Saver' op to save and restore all the variables
saverAutoencoder = tf.train.Saver([W_ae0, b_ae0, W_ae1, b_ae1, W_ae2, b_ae2, W_ae3, b_ae3, W_output, b_output])
saver = tf.train.Saver()
model_ae = "model/dal_ae.ckpt"
model_path = "model/dal.ckpt"

# Restore only the layers up to fc1 (included)
# Calling function `init_fn(sess)` will load all the pretrained weights.
variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=None)
init_fn = tf.contrib.framework.assign_from_checkpoint_fn(model_ae, variables_to_restore)

# Merge all the summaries and write them out
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('log/dal_train', flush_secs=10)
val_writer = tf.summary.FileWriter('log/dal_val', flush_secs=10)

print("Result with timesteps = " + str(timesteps) + ", predict timesteps = " + str(pred_timesteps) + "\n")

# Soft placement allows placing on CPU ops without GPU implementation.
session_config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False)
session_config.gpu_options.per_process_gpu_memory_fraction = 0.8
session_config.gpu_options.visible_device_list = '2,3'

# Start training
with tf.Session(config=session_config) as sess:
    # Run the initializer
    sess.run(init)

    if is_training_ae:
        print('Training Autoencoder...')
        for step in range(1, training_steps+1):
            # Make the training batch for each step
            batch_x, batch_y = batch_creator_ae(X_train, batch_size, X_train.shape[0])

            # Run optimization
            _, train_loss = sess.run([optimizer_ae, loss_ae], feed_dict = {x: batch_x, y_ae: batch_y})
            #train_writer.add_summary(summary, step)

            # compute error on validate set
            batch_x, batch_y = batch_creator_ae(X_val, batch_size, X_val.shape[0])

            [validate_loss] = sess.run([loss_ae], feed_dict={x: batch_x, y_ae: batch_y})
            #val_writer.add_summary(summary, step)

            # Print result
            if step % display_step == 0 or step == 1:
                print("Iter = " + str(step) + ". Train error = {:.6f}. Validate error = {:.6f}".format(train_loss, validate_loss))
                    
        print("Training Finished!")
        # Save model weights to disk
        save_path = saver.save(sess, model_ae)
        print("Model saved in file: %s" % save_path)
        exit() # finish AutoEncoder training

with tf.Session(config=session_config) as sess:
    # Run the initializer
    sess.run(init)
    if is_training_all:
        print('Training DAL...') # Training DAL
        saverAutoencoder.restore(sess, model_ae)

        for step in range(1, training_steps+1):
            # Make the training batch for each step
            batch_x, batch_y, batch_ymap = batch_creator_all(X_train, batch_size, X_train.shape[0])

            # Run optimization
            _, train_loss, summary = sess.run([optimizer, loss, merged], feed_dict = {x: batch_x, y: batch_y, y_map: batch_ymap})
            #train_writer.add_summary(summary, step)

            # compute error on validate set
            batch_x, batch_y, batch_ymap = batch_creator_all(X_val, batch_size, X_val.shape[0])

            [validate_loss, summary] = sess.run([loss, merged], feed_dict={x: batch_x, y: batch_y, y_map: batch_ymap})
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
    var1 = list()
    var2 = list()
    chi = list()
    for i in range(test_steps):
        batch_x, batch_y, batch_ymap = batch_creator_all(X_test, batch_size, X_test.shape[0])
        
        start_time = time.time()
        [out_test] = sess.run([output_spa], feed_dict={x: batch_x, y: batch_y, y_map: batch_ymap})

        inv_out = scaler.inverse_transform(out_test.flatten().reshape(-1, 1))
        inv_yhat = np.multiply(inv_out, batch_ymap.flatten().reshape(-1, 1))
        inv_y = scaler.inverse_transform(batch_y.flatten().reshape(-1, 1))
        loss_value = sqrt(mean_squared_error(inv_y, inv_yhat) * loss_ratio)

        from scipy import stats
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
            ax.imshow(X_i1, cmap='gray', interpolation='none')

            Xa = list()
            X_i1 = X_i1.flatten()
            for s in range(len(station_map)):
                if X_i1[station_map[s]] > 0:
                    Xa.append(X_i1[station_map[s]])
            chi.append(float(stats.chisquare(X_i1.flatten(), X_i.flatten())[0]))
            var1.append(float(np.var(X_i)))
            var2.append(float(np.var(X_i1)))'''

            #plt.show()

        elapsed_time += time.time() - start_time
        loss_test += loss_value

    # Print validate error
    print("Test Error = {:.6f}. Elapsed time = {:.3f}".format(loss_test/test_steps, elapsed_time/test_steps))

    output_path = "output/dal.txt"
    outfile = open(output_path, 'a')
    outfile.write('\n')
    outfile.write("Result with timesteps = " + str(timesteps) + ", predict timesteps = " + str(pred_timesteps) + "\n")
    outfile.write("Test Error = {:.6f}. Elapsed time = {:.3f}\n".format(loss_test/test_steps, elapsed_time/test_steps))

