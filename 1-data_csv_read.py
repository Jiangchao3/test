# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 16:00:45 2021

@author: admin
"""
import pandas as pd
import numpy as np
import copy
from numpy import multiply, cos, arcsin, pi
import os

filepath = 'H:/QJC/9-MODEL/TC/IBTrACS/ibtracs.WP.list.v04r00.csv'

# read in full data of west northern pacific
data_full = pd.read_csv(filepath,sep=',', header=0, na_values=' ').drop(index=0)

data_full['SEASON'] = data_full['SEASON'].astype('int')

data_sliced = data_full.loc[(data_full['SEASON']>1956) & (data_full['SEASON']<2019)]

data_sliced.dropna(subset=['USA_ATCF_ID'],how='any',inplace=True)

data_sliced['USA_LAT'] = data_sliced['USA_LAT'].astype('float')
data_sliced['USA_LON'] = data_sliced['USA_LON'].astype('float')

#ID_size_DF  = pd.DataFrame([i[1].shape[0] for i in data_sliced.groupby('USA_ATCF_ID')], index=[i[0] for i in data_sliced.groupby('USA_ATCF_ID')])

ID_size_array = np.array([[i[0],i[1].shape[0]] for i in data_sliced.groupby('USA_ATCF_ID')])

lon = np.full([len(ID_size_array),360], np.nan)
lat = np.full([len(ID_size_array),360], np.nan)

lon  = np.array([list(i[1]['USA_LON'].values)+[np.nan for j in range(360-len(i[1]['USA_LON'].values))] for i in data_sliced.groupby('USA_ATCF_ID')])
lat  = np.array([list(i[1]['USA_LAT'].values)+[np.nan for j in range(360-len(i[1]['USA_LAT'].values))] for i in data_sliced.groupby('USA_ATCF_ID')])
wind = np.array([list(i[1]['USA_WIND'].values)+[np.nan for j in range(360-len(i[1]['USA_WIND'].values))] for i in data_sliced.groupby('USA_ATCF_ID')])
time = np.array([list(i[1]['ISO_TIME'].values)+[np.nan for j in range(360-len(i[1]['ISO_TIME'].values))] for i in data_sliced.groupby('USA_ATCF_ID')])

# lat and lon of macao
lat_macao = np.mat(22.2*np.ones(lon.shape))
lon_macao = np.mat(113.55*np.ones(lon.shape))

# use haversine-formula to calculate the radius from macau less than 400 km
# https://stackoverflow.com/questions/27928/calculate-distance-distance_caletween-two-latitude-longitude-points-haversine-formula
def distance_mtx(lat1_mtx, lon1_mtx, lat2_mtx, lon2_mtx):
    p = pi/180
    a = 0.5 - cos((lat2_mtx-lat1_mtx)*p)/2 + multiply(multiply(cos(lat1_mtx*p),cos(lat2_mtx*p)),(1-cos((lon2_mtx-lon1_mtx)*p)))/2
    a_sqrt = np.sqrt(a)
    R = 2*6371*arcsin(a_sqrt)  #2*R*asin...
    return R
distance = distance_mtx(lat_macao, lon_macao, lat, lon)

# make sure if the distance less than 200 km
threshold_dis = 200
distance_cal = np.where(distance < threshold_dis,1,distance) 
distance_cal = np.where(distance_cal > threshold_dis,0,distance_cal)
distance_cal[np.isnan(distance_cal)]=0
points_count_dis = np.sum(distance_cal,axis=1)
where_position_dis = list(np.nonzero(points_count_dis)[0])

# make sure if this TC is a typhoon or not
threshold_typhoon = 63
wind_cal = wind
wind_cal = np.where(wind_cal < threshold_typhoon,0,wind_cal)
wind_cal = np.where(wind_cal > threshold_typhoon,1,wind_cal)
wind_cal[np.isnan(wind_cal)]=0
points_count_typhoon = np.sum(wind_cal,axis=1)

# make sure the event is selected by dis and typhoon wind
result_dis_typhoon = points_count_dis*points_count_typhoon
where_positions_dis_typhoon = list(np.nonzero(result_dis_typhoon)[0])

where_positions_dis_typhoon_deleted = copy.deepcopy(where_positions_dis_typhoon)

# We first plot all the 85 tracks, 9 tracks below do not meet our requirements
# so we delete them manauly 
where_positions_dis_typhoon_deleted.remove(np.argwhere(ID_size_array[:,0]=='WP141957')[0][0])
where_positions_dis_typhoon_deleted.remove(np.argwhere(ID_size_array[:,0]=='WP221964')[0][0])
where_positions_dis_typhoon_deleted.remove(np.argwhere(ID_size_array[:,0]=='WP161970')[0][0])
where_positions_dis_typhoon_deleted.remove(np.argwhere(ID_size_array[:,0]=='WP131986')[0][0])
where_positions_dis_typhoon_deleted.remove(np.argwhere(ID_size_array[:,0]=='WP191986')[0][0])
where_positions_dis_typhoon_deleted.remove(np.argwhere(ID_size_array[:,0]=='WP121991')[0][0])
where_positions_dis_typhoon_deleted.remove(np.argwhere(ID_size_array[:,0]=='WP081995')[0][0])
where_positions_dis_typhoon_deleted.remove(np.argwhere(ID_size_array[:,0]=='WP241995')[0][0])
where_positions_dis_typhoon_deleted.remove(np.argwhere(ID_size_array[:,0]=='WP211999')[0][0])

ID_by_dis_typhoon         = ID_size_array[where_positions_dis_typhoon,0].tolist()
ID_by_dis_typhoon_deleted = ID_size_array[where_positions_dis_typhoon_deleted,0].tolist()

# sort the selected typhoon ID
ID_by_dis_typhoon_deleted_sort = [ID_by_dis_typhoon_deleted[i][0:2]+\
                                  ID_by_dis_typhoon_deleted[i][4:8]+\
                                  ID_by_dis_typhoon_deleted[i][2:4] for i in range(len(ID_by_dis_typhoon_deleted))]  
    
ID_by_dis_typhoon_deleted_sort.sort()

ID_by_dis_typhoon_deleted_sort = [ID_by_dis_typhoon_deleted_sort[i][0:2]+\
                                  ID_by_dis_typhoon_deleted_sort[i][6:8]+\
                                  ID_by_dis_typhoon_deleted_sort[i][2:6] for i in range(len(ID_by_dis_typhoon_deleted_sort))] 

# save the first ID file (do not classify to north and west forward path)
(pd.DataFrame(ID_by_dis_typhoon_deleted_sort)).to_csv('1-ID_by_dis_typhoon_deleted_sort.csv')

# Classify the typhoon's track to two different types: north forward path and west forward path
# We firstly plot the total 76 tracks using package tropycal and the ID in file of 1-ID_by_dis_typhoon_deleted_sort.csv
# And then complete the classification by observing the tracks of each TC event 
# for north forward path 
dir_trackplot_north_forward = 'H:/QJC/9-MODEL/TC/IBTrACS/trackplot/north_forward' 
trackplot_north_forward = []
# read the filename of each plot
for inputfile in os.listdir(dir_trackplot_north_forward):
    fileName = os.path.splitext(inputfile)[0]
    trackplot_north_forward.append(fileName)           
    print(fileName)
# reverse the order of year and number for each track
trackplot_north_forward_sort = [trackplot_north_forward[i][0:2]+\
                                trackplot_north_forward[i][4:8]+\
                                trackplot_north_forward[i][2:4] for i in range(len(trackplot_north_forward))]  
# rearrange the track 
trackplot_north_forward_sort.sort()
# reverse the order of the number and year for each track
trackplot_north_forward_sort = [trackplot_north_forward_sort[i][0:2]+\
                                trackplot_north_forward_sort[i][6:8]+\
                                trackplot_north_forward_sort[i][2:6] for i in range(len(trackplot_north_forward_sort))] 
# for west forward path
dir_trackplot_west_forward  = 'H:/QJC/9-MODEL/TC/IBTrACS/trackplot/west_forward'            
trackplot_west_forward  = []  
# read the filename of each plot
for inputfile in os.listdir(dir_trackplot_west_forward):
    fileName = os.path.splitext(inputfile)[0]
    trackplot_west_forward.append(fileName)           
    print(fileName)
# reverse the order of year and number for each track   
trackplot_west_forward_sort = [trackplot_west_forward[i][0:2]+\
                               trackplot_west_forward[i][4:8]+\
                               trackplot_west_forward[i][2:4] for i in range(len(trackplot_west_forward))]  
# rearrange the track
trackplot_west_forward_sort.sort()
# reverse the order of the number and year for each track
trackplot_west_forward_sort = [trackplot_west_forward_sort[i][0:2]+\
                               trackplot_west_forward_sort[i][6:8]+\
                               trackplot_west_forward_sort[i][2:6] for i in range(len(trackplot_west_forward_sort))] 

# save data to two different files
(pd.DataFrame(trackplot_north_forward_sort)).to_csv('1-ID_trackplot_north_forward_sort.csv')   
(pd.DataFrame(trackplot_west_forward_sort)).to_csv('1-ID-trackplot_west_forward_sort.csv')

# the coresponding Sequence number in ID_size_array
Se_trackplot_north_forward_sort = [np.argwhere(ID_size_array[:,0]==trackplot_north_forward_sort[i])[0][0] for i in range(len(trackplot_north_forward_sort))]
Se_trackplot_west_forward_sort  = [np.argwhere(ID_size_array[:,0]==trackplot_west_forward_sort[i])[0][0] for i in range(len(trackplot_west_forward_sort))]

# function to calculate the values of max wind speed and the corresponding distance in radius
def values_max_wind_min_dis(Se_trackplot_sort):
    max_wind_in_radius = []
    min_dis_in_radius  = []
    for i in range(len(Se_trackplot_sort)): 
        x = Se_trackplot_sort[i]
        # return the index of all points within radius
        idx_in_radius = np.argwhere(distance_cal[x]==1) 
        # return the wind speed series of all points within radius
        idx_wind_in_radius = pd.DataFrame(wind[x]).loc[idx_in_radius.ravel()]
        # return the distance series of all points within radius
        idx_distance_in_radius = pd.DataFrame(np.array(distance[x]).T).loc[idx_in_radius.ravel()]
        # find the index of all maxmimum wind speed points, maybe there is one point, or multiple points
        idx_max_wind_in_radius = idx_wind_in_radius[idx_wind_in_radius[0]==idx_wind_in_radius[0].max()].index.to_list()
        # return the index of the point with minimum distance and maximum wind speed
        idx_max_wind_min_distance = idx_distance_in_radius.loc[idx_max_wind_in_radius][0].idxmin()
        temp_max_wind = idx_wind_in_radius[0].loc[idx_max_wind_min_distance]
        temp_min_dis  = idx_distance_in_radius[0].loc[idx_max_wind_min_distance]
        max_wind_in_radius.append(temp_max_wind)
        min_dis_in_radius.append(temp_min_dis)
    return (max_wind_in_radius,min_dis_in_radius)

# calculate the values of max wind speed and the corresponding distance in radius for north and west forward, seperately
(max_wind_north_forward,min_dis_north_forward)=values_max_wind_min_dis(Se_trackplot_north_forward_sort)
(max_wind_west_forward,min_dis_west_forward)=values_max_wind_min_dis(Se_trackplot_west_forward_sort)

(pd.DataFrame(max_wind_north_forward)).to_csv('1-value_max_wind_north.csv')
(pd.DataFrame(min_dis_north_forward)).to_csv('1-value_min_dis_north.csv') 
(pd.DataFrame(max_wind_west_forward)).to_csv('1-value_max_wind_west.csv')
(pd.DataFrame(min_dis_west_forward)).to_csv('1-value_min_dis_west.csv') 
    
# define a function to calculate the corresponding time of TC track point with max wind speed
# and min distance to a certain place (here is macao), and the before 2 days and next 2 days 
def times_max_wind_min_dis_five_days(Se_trackplot_sort):
    time_max_wind_min_dis_in_radius = []
    time_five_days_in_radius = []
    for i in range(len(Se_trackplot_sort)): 
        temp_list = []
        x = Se_trackplot_sort[i]
        # return the index of all points within radius
        idx_in_radius = np.argwhere(distance_cal[x]==1) 
        # return the wind speed series of all points within radius
        idx_wind_in_radius = pd.DataFrame(wind[x]).loc[idx_in_radius.ravel()]
        # return the distance series of all points within radius
        idx_distance_in_radius = pd.DataFrame(np.array(distance[x]).T).loc[idx_in_radius.ravel()]
        # find the index of all maxmimum wind speed points, maybe there is one point, or multiple points
        idx_max_wind_in_radius = idx_wind_in_radius[idx_wind_in_radius[0]==idx_wind_in_radius[0].max()].index.to_list()
        # return the index of the point with minimum distance and maximum wind speed
        idx_max_wind_min_distance = idx_distance_in_radius.loc[idx_max_wind_in_radius][0].idxmin()
        # returen the time of the point with minimum distance and maximum wind speed
        temp_time = time[x,idx_max_wind_min_distance]
        # transform UTC time to Beijing time (+8 h)
        temp_time_datetime_now = pd.to_datetime(temp_time)+ pd.to_timedelta('8 h')
        # calculate the successive five days time
        temp_time_datetime_next1 = copy.deepcopy(temp_time_datetime_now) + pd.to_timedelta('24 h')
        temp_time_datetime_next2 = copy.deepcopy(temp_time_datetime_now) + pd.to_timedelta('48 h')
        temp_time_datetime_before1  = copy.deepcopy(temp_time_datetime_now) - pd.to_timedelta('24 h')
        temp_time_datetime_before2  = copy.deepcopy(temp_time_datetime_now) - pd.to_timedelta('48 h')
        # save to temp list
        temp_list = [temp_time_datetime_before2,temp_time_datetime_before1,temp_time_datetime_now,temp_time_datetime_next1,temp_time_datetime_next2]
        time_max_wind_min_dis_in_radius.append(temp_time_datetime_now)
        time_five_days_in_radius.append(temp_list)
    return (time_max_wind_min_dis_in_radius,time_five_days_in_radius)

# calculate the time of TC point with max wind speed and min distance to macao, and the before 2 days and next two days
(north_time_max_wind_min_dis_in_radius, north_time_five_days_in_radius) = times_max_wind_min_dis_five_days(Se_trackplot_north_forward_sort)
(west_time_max_wind_min_dis_in_radius, west_time_five_days_in_radius)   = times_max_wind_min_dis_five_days(Se_trackplot_west_forward_sort)

# concatenate the time of five days 
north_time_five_days_in_radius_concatente = copy.deepcopy(list(np.concatenate(north_time_five_days_in_radius)))
west_time_five_days_in_radius_concatente  = copy.deepcopy(list(np.concatenate(west_time_five_days_in_radius)))

# save data to csv files
(pd.DataFrame(north_time_max_wind_min_dis_in_radius)).to_csv('1-north_time_max_wind_min_dis_in_radius.csv')

(pd.DataFrame(west_time_max_wind_min_dis_in_radius)).to_csv('1-west_time_max_wind_min_dis_in_radius.csv')

(pd.DataFrame(north_time_five_days_in_radius_concatente)).to_csv('1-north_time_five_days_in_radius_concatente.csv')

(pd.DataFrame(west_time_five_days_in_radius_concatente)).to_csv('1-west_time_five_days_in_radius_concatente.csv')

# transform time for each days 00:00:00
north_time_max_wind_min_dis_in_radius_00 = [i-pd.to_timedelta('%s h'%i.hour) for i in north_time_max_wind_min_dis_in_radius]

west_time_max_wind_min_dis_in_radius_00  = [i-pd.to_timedelta('%s h'%i.hour) for i in west_time_max_wind_min_dis_in_radius]

north_time_five_days_in_radius_concatente_00 = [i-pd.to_timedelta('%s h'%i.hour) for i in north_time_five_days_in_radius_concatente]

west_time_five_days_in_radius_concatente_00  = [i-pd.to_timedelta('%s h'%i.hour) for i in west_time_five_days_in_radius_concatente]

# save data to csv files 
(pd.DataFrame(north_time_max_wind_min_dis_in_radius_00)).to_csv('1-north_time_max_wind_min_dis_in_radius_00.csv')

(pd.DataFrame(west_time_max_wind_min_dis_in_radius_00)).to_csv('1-west_time_max_wind_min_dis_in_radius_00.csv')

(pd.DataFrame(north_time_five_days_in_radius_concatente_00)).to_csv('1-north_time_five_days_in_radius_concatente_00.csv')

(pd.DataFrame(west_time_five_days_in_radius_concatente_00)).to_csv('1-west_time_five_days_in_radius_concatente_00.csv')




