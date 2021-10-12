#!/usr/bin/env python2

import numpy as np
import pandas as pd
import h5py
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class read_data(object):

    def __init__(self):
        # grid size
        self.grid_size = 1024
        # number of pollution stations
        self.poll_station = 37
        # number of meteorology stations
        self.met_station = 28
        # array of wind direction
        self.arr=["N","NNE","NE","ENE","E","ESE", "SE", "SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"]

    def read_pollution_data(self, data_file, col_idx):
        df = pd.read_csv(data_file)
        data = df.values
        
        # create grid of data
        grid_size = self.grid_size
        no_station = self.poll_station
        station_map = data[0:no_station, 1].astype(int)
        X = list()
        X1 = np.zeros(grid_size)
        for i in range(len(data)):
            map_idx = int(data[i,1])
            val = data[i, col_idx]
            if val != 'null':
                X1[map_idx] = float(val)
            if (i+1) % no_station == 0:
                X.append(X1)
                X1 = np.zeros(grid_size)

        return np.array(X), np.array(station_map)

    def save_pollution_data(self, pollution_file, col_idx=6):
        # pollution data path
        data_path = '../seoul_data/'
        # read data
        X1,_ = self.read_pollution_data(data_path + 'grid_pollution2015.csv', col_idx)
        X2,_ = self.read_pollution_data(data_path + 'grid_pollution2016.csv', col_idx)
        X3,station_map = self.read_pollution_data(data_path + 'grid_pollution2017.csv', col_idx)
        X = np.concatenate([X1, X2, X3], axis = 0)

        with h5py.File(pollution_file, 'w') as hf:
            hf.create_dataset("pollution", data=X)
            hf.create_dataset("station_map", data=station_map)

    def get_float(self, val):
        if val != 'null':
            return float(val)
        else:
            return None

    def find_nearest_station(self, e, station_map):
        # get col, row of cell e
        row = int(e/32)
        col = e - row*32
        # compute min distance with a station in station_map
        d, d_min = 1024, 1024
        nearest = 0
        for t in range(len(station_map)):
            s = station_map[t]
            i = int(s/32)
            j = s - i*32
            d = abs(row-i) + abs(col-j)
            if d < d_min:
                d_min = d
                nearest = s
        return nearest

    def angToDir(self, deg):
        val=int((deg/22.5)+.5)
        return self.arr[(val % 16)]

    # read and pre-process meteorology data
    def read_meteorology_data(self, data_file):
        df = pd.read_csv(data_file)
        data = df.values
        
        # create grid of data
        grid_size = self.grid_size
        no_station = self.met_station
        station_map = data[0:no_station, 13].astype(int)

        X = list()
        X1 = np.zeros((grid_size, 6))
        print(data_file)
        
        # find nearest station
        X_nearest = list()
        for j in range(X1.shape[0]):
            nearest = self.find_nearest_station(j, station_map)
            X_nearest.append(nearest)

        start_hour = int(data[0,15])
        for i in range(len(data)):
            # get value from data
            map_idx = int(data[i,13])
            temp = self.get_float(data[i,4])
            wind_dir = self.get_float(data[i,5])
            wind_spd = self.get_float(data[i,6])
            rain = self.get_float(data[i,7])
            humid = self.get_float(data[i,10])
            pres_min = self.get_float(data[i,8])
            pres_max = self.get_float(data[i,9])
            pres = None
            if pres_min != None and pres_max != None:
                pres = (pres_min + pres_max) / 2

            X1[map_idx][0] = temp
            X1[map_idx][1] = wind_dir
            X1[map_idx][2] = wind_spd
            X1[map_idx][3] = rain
            X1[map_idx][4] = humid
            X1[map_idx][5] = pres
            
            if (i+1 < len(data)):
                hour = int(data[i+1,15])
            if hour > start_hour or i+1 == len(data):
                start_hour = hour
                # update value of each cells by nearest station
                for j in range(X1.shape[0]):
                    X1[j] = X1[X_nearest[j]]
                # update nan value in X1 by mean value
                df = pd.DataFrame(data=X1)
                df = df.fillna(df.mean())
                # if all values are null => replace by previous values
                for t in range(6):
                    if (np.isnan(df.ix[:,t]).all()):
                        df.ix[:,t] = X[len(X)-1][:,t]
                   
                X.append(df.values)
                X1 = np.zeros((grid_size, 6))

        return np.array(X), np.array(station_map)

    def save_meteorology_data(self, met_file):
        data_path = '../seoul_data/'
        # read data
        X1,_ = self.read_meteorology_data(data_path + 'grid_weather2015.csv')
        print(X1.shape)
        X2,_ = self.read_meteorology_data(data_path + 'grid_weather2016.csv')
        print(X2.shape)
        X3,station_map = self.read_meteorology_data(data_path + 'grid_weather2017.csv')
        X_met = np.concatenate([X1, X2, X3], axis = 0)
        print(X_met.shape)

        with h5py.File(met_file, 'w') as hf:
            hf.create_dataset("met", data=X_met)
            hf.create_dataset("station_map", data=station_map)

    def save_meteorology_numpy(self, met_file_npy):
        met_file = 'data/meteorology.h5'
        with h5py.File(met_file, 'r') as hf:
            X_met = hf['met'][:]
        print(X_met.shape)
        # processing met data
        df = pd.DataFrame(data=X_met.reshape(X_met.shape[0]*X_met.shape[1], X_met.shape[2]))

        # convert wind direction angle
        def angToDir(deg):
            val=int((deg/22.5)+.5)
            arr=["N","NNE","NE","ENE","E","ESE", "SE", "SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"]
            return arr[(val % 16)]
        v = np.vectorize(angToDir)

        wind_direction = v(df.ix[:,1])
        temp = df.ix[:,0].values.reshape(-1,1)
        wind_speed = df.ix[:,2].values.reshape(-1,1)
        rain = df.ix[:,3].values.reshape(-1,1)
        humid = df.ix[:,4].values.reshape(-1,1)
        pres = df.ix[:,5].values.reshape(-1,1)

        # Scaler
        scaler = MinMaxScaler(feature_range=(0, 1))

        temp = scaler.fit_transform(temp)
        rain = scaler.fit_transform(rain)
        wind_speed = scaler.fit_transform(wind_speed)
        humid = scaler.fit_transform(humid)
        pres = scaler.fit_transform(pres)

        # Convert to OneHotEncoding
        le = LabelEncoder().fit(wind_direction.ravel())
        enc = OneHotEncoder(sparse=False)
        wind_direction = enc.fit_transform(le.transform(wind_direction).reshape(-1,1))

        # Weather data concatenation
        X_met_concat = np.concatenate([temp, rain, wind_speed, wind_direction, humid, pres], axis=1)
        met_features = X_met_concat.shape[1]
        X_met_concat = X_met_concat.reshape(X_met.shape[0], X_met.shape[1], met_features)
        print('Weather data shape: {}'.format(X_met_concat.shape))
        np.save(met_file_npy, X_met_concat)

    def find_nearest_traffic_station(self, e, station_map):
        # get col, row of cell e
        row = int(e/32)
        col = e - row*32
        # compute min distance with a station in station_map
        d, d_min = 1024, 1024
        for t in range(len(station_map)):
            s = station_map[t]
            i = int(s/32)
            j = s - i*32
            d = abs(row-i) + abs(col-j)
            if d < d_min:
                d_min = d
        # find all possible nearest stations based on found d_min
        nearest = list()
        for t in range(len(station_map)):
            s = station_map[t]
            i = int(s/32)
            j = s - i*32
            d = abs(row-i) + abs(col-j)
            if d == d_min:
                nearest.append(s)

        return nearest

    # read and pre-process traffic data
    def read_traffic_data(self, data_file):
        df = pd.read_csv(data_file)
        
        # create grid of data
        grid_size = self.grid_size
        X = list()

        hour_list = df.hour.unique()
        print(data_file, hour_list)
        num_hour = np.amax(hour_list)
        for h in range(num_hour, num_hour+1):
            data = df.loc[df.hour == h]
            if data.empty:
                # set X1 to zero
                X1 = np.zeros(grid_size)
                X.append(X1)
                continue
                
            station_map = data.loc[data.value != 'null']['map_idx'].values
            if (len(station_map) == 0):
                # set X1 to zero
                X1 = np.zeros(grid_size)
                X.append(X1)
                continue
                
            dict_value = {}
            for i in range(len(station_map)):
                dict_value[station_map[i]] = float(data.loc[data.map_idx == station_map[i]]['value'])
            if h % 100 == 0:
                print(h)
            # init grid cell values
            X1 = np.zeros(grid_size)
            # update grid values by nearest stations
            for i in range(grid_size):
                nearest = self.find_nearest_traffic_station(i, station_map)
                avg = 0 # average value of all nearest stations
                for j in range(len(nearest)):
                    avg += dict_value[nearest[j]]
                avg = avg/len(nearest)
                X1[i] = avg

            X.append(X1)
        
        return np.array(X)

    def save_traffic_data(self, traffic_file):
        data_path = '../seoul_data/'
        # read data
        X1 = self.read_traffic_data(data_path + 'grid_traffic2015.csv')
        X2 = self.read_traffic_data(data_path + 'grid_traffic2016.csv')
        X3 = self.read_traffic_data(data_path + 'grid_traffic2017.csv')
        X = np.concatenate([X1, X2, X3], axis = 0)
        print(X.shape)

        with h5py.File(traffic_file, 'w') as hf:
            hf.create_dataset("traffic", data=X)

    # read and pre-process speed data
    def read_speed_data(self, data_file):
        df = pd.read_csv(data_file)
        
        # create grid of data
        grid_size = self.grid_size
        X = list()

        hour_list = df.hour.unique()
        print(data_file, hour_list)
        num_hour = np.amax(hour_list)
        for h in range(num_hour+1):
            data = df.loc[df.hour == h]
            if data.empty:
                #print(h)
                # set X1 to zero
                X1 = np.zeros(grid_size)
                X.append(X1)
                continue
                
            station_map = data['map_idx'].values
                
            # init grid cell values
            X1 = np.zeros(grid_size)
            # update grid values
            for i in range(len(station_map)):
                if station_map[i] < grid_size:
                    X1[station_map[i]] = float(data.iloc[i,2])

            X.append(X1)
            if h % 100 == 0:
                print(h)
        
        return np.array(X)

    def save_speed_data(self, speed_file):
        data_path = '../seoul_data/'
        # read data
        X1 = self.read_speed_data(data_path + 'grid_speed2015.csv')
        print(X1.shape)
        X2 = self.read_speed_data(data_path + 'grid_speed2016.csv')
        print(X2.shape)
        X3 = self.read_speed_data(data_path + 'grid_speed2017.csv')
        print(X3.shape)
        X = np.concatenate([X1, X2, X3], axis = 0)
        print(X.shape)

        with h5py.File(speed_file, 'w') as hf:
            hf.create_dataset("speed", data=X)

    # read and pre-process china data
    def read_china_data(self, data_file):
        df = pd.read_csv(data_file)
        X = list()

        print(data_file)
        num_hour = (365+366+365)*24 # 2015, 2016, 2017
        for h in range(num_hour):
            data = df.loc[df.hour == h]
            if data.empty:
                # insert zero
                X.append(0)
            else:
                try:
                    X.append(float(data['PM25']))
                except:
                    print(h)
        
        return np.array(X)

    def save_china_data(self, china_file):
        data_path = '../aqi_data/aqi_china/berkeley_earth/'
        # read data
        X1 = self.read_china_data(data_path + 'Beijing.csv')
        print(X1.shape)
        X2 = self.read_china_data(data_path + 'Shanghai.csv')
        print(X2.shape)
        X3 = self.read_china_data(data_path + 'Shandong.csv')
        print(X3.shape)

        with h5py.File(china_file, 'w') as hf:
            hf.create_dataset("beijing", data=X1)
            hf.create_dataset("shanghai", data=X2)
            hf.create_dataset("shandong", data=X3)

    # read and pre-process daegu data
    def read_daegu_data(self, data_file):
        N = 1024 # number of cells
        timerange_init = -1
        X = list()
        station_map = np.zeros(N)
        # read pm2_5 values
        with open(data_file, 'r') as f:
            next(f) # by pass the header
            for row in f:
                res = row.split(',')
                timerange, map_idx, pm2_5 = int(res[0]), int(res[1]), float(res[2])
                # create list of pm2_5
                if (timerange != timerange_init):
                    x1 = list()
                    for j in range(N):
                        x1.append(0)
                    timerange_init = timerange
                    X.append(x1)
                else:
                    x1 = X[timerange]
                x1[map_idx] = pm2_5
                station_map[map_idx] = 1

        X = np.asarray(X)
        station_idx = []
        for i in range(N):
            if station_map[i] == 1:
                station_idx.append(i)
        seed = 1
        rng = np.random.RandomState(seed)
        station_map = rng.choice(station_idx, 15)
        print(station_map)

        return X, station_map

    def save_daegu_data(self, daegu_file):
        data_path = '../analysis_v1/data/'
        # read data
        X1,_ = self.read_daegu_data(data_path + 'daegu_full_09_2017.csv')
        print(X1.shape)
        X2,_ = self.read_daegu_data(data_path + 'daegu_full_10_2017.csv')
        print(X2.shape)
        X3,_ = self.read_daegu_data(data_path + 'daegu_full_11_2017.csv')
        print(X3.shape)
        X4, station_map = self.read_daegu_data(data_path + 'daegu_full_12_2017.csv')
        print(X4.shape)
        X = np.concatenate([X1, X2, X3, X4], axis = 0)
        print(X.shape)

        with h5py.File(daegu_file, 'w') as hf:
            hf.create_dataset("daegu", data=X)
            hf.create_dataset("station_map", data=station_map)

    def plot_data(self):
        # pollution
        with h5py.File('data/pollutionPM25.h5', 'r') as hf:
            X_pol = hf['pollution'][:]
        print(X_pol.shape)
        
        # traffic
        with h5py.File('data/traffic.h5', 'r') as hf:
            X_tf = hf['traffic'][:]
        print(X_tf.shape)

        # speed
        with h5py.File('data/speed.h5', 'r') as hf:
            X_sp = hf['speed'][:]
        print(X_sp.shape)

        # meteorology
        with h5py.File('data/meteorology.h5', 'r') as hf:
            X_met = hf['met'][:]
        print(X_met.shape)

        # plot image
        i = 1000
        X = X_met[i,:,0].reshape(32, 32)
        plt.imshow(X, cmap='gray', interpolation='none')

        '''import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(1, 3)
        gs.update(wspace=0.025, hspace=0.05)

        fig = plt.figure()
        ax = fig.add_subplot(gs[0,0])
        i = 0
        ax.set_title('Pollution hour = {}'.format(i))
        X1 = X_pol[i].reshape(32,32)
        ax.imshow(X1, cmap='gray', interpolation='none')

        ax = fig.add_subplot(gs[0,1])
        ax.set_title('Pollution hour = {}'.format(i+1))
        X1 = X_pol[i+1].reshape(32,32)
        #X1 = X_met[i,:,0].reshape(32,32)
        ax.imshow(X1, cmap='gray', interpolation='none')

        ax = fig.add_subplot(gs[0,2])
        ax.set_title('Pollution hour = {}'.format(i+2))
        X1 = X_pol[i+2].reshape(32,32)
        ax.imshow(X1, cmap='gray', interpolation='none')'''

        '''fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        text1 = ax1.text(0.2, 0.95, '', transform=ax1.transAxes, bbox=dict(facecolor='red', alpha=0.5))
        text2 = ax2.text(0.2, 0.95, '', transform=ax2.transAxes, bbox=dict(facecolor='red', alpha=0.5))

        import time
        def animate(f, i):
            i = i+400+f
            # f is the next frame of animation
            text1.set_text('Speed at hour = {}'.format(i))
            X1 = X_sp[i].reshape(32,32)
            im1 = ax1.imshow(X1, cmap='gray', interpolation='none')

            text2.set_text('Traffic at hour = {}'.format(i))
            X1 = X_tf[i].reshape(32,32)
            im2 = ax2.imshow(X1, cmap='gray', interpolation='none')

            return im1, im2, text1, text2

        i = 0
        ani = animation.FuncAnimation(fig, animate, interval=1000, repeat=False, blit=True, fargs=(i,))'''

        #mng = plt.get_current_fig_manager()
        #mng.resize(*mng.window.maxsize())
        plt.show()

# test
if __name__ == '__main__':
    r = read_data()
    #r.save_daegu_data('data/daegu.h5')
    r.plot_data()
    
    
