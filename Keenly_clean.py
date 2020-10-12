# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 10:03:56 2020

@author: jwu02
"""


import os
path="D:\Dropbox\OMSA\ISYEPractice\Keenly\Data"
os.chdir(path)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date


# read the sensor data
df0 = pd.read_csv('sensor_data_gtv2000000000000.gzip', compression='gzip',  sep=',', quotechar='"')
df1 = pd.read_csv('sensor_data_gtv2000000000001.gzip', compression='gzip',  sep=',', quotechar='"')
df2 = pd.read_csv('sensor_data_gtv2000000000002.gzip', compression='gzip',  sep=',', quotechar='"')
df3 = pd.read_csv('sensor_data_gtv2000000000003.gzip', compression='gzip',  sep=',', quotechar='"')
df4 = pd.read_csv('sensor_data_gtv2000000000004.gzip', compression='gzip',  sep=',', quotechar='"')

df5 = pd.read_csv('sensor_data_gtv2000000000005.gzip', compression='gzip',  sep=',', quotechar='"')
df6 = pd.read_csv('sensor_data_gtv2000000000006.gzip', compression='gzip',  sep=',', quotechar='"')
df7 = pd.read_csv('sensor_data_gtv2000000000007.gzip', compression='gzip',  sep=',', quotechar='"')
df8 = pd.read_csv('sensor_data_gtv2000000000008.gzip', compression='gzip',  sep=',', quotechar='"')
df9 = pd.read_csv('sensor_data_gtv2000000000009.gzip', compression='gzip',  sep=',', quotechar='"')


# Concat dataset
df04 =pd.concat([df0, df1, df2, df3, df4])

df59 =pd.concat([df5, df6, df7, df8, df9])


# select key variables
df04a=df04.loc[:,['Timestamp', 'StatusCode',  'Distance', 'SignalQuality', 'RespirationRate',  'DeviceID', 'DetectionCount','PatientID', 'thermalPresence']]


df59a=df59.loc[:,['Timestamp', 'StatusCode',  'Distance', 'SignalQuality', 'RespirationRate',  'DeviceID', 'DetectionCount','PatientID', 'thermalPresence']]

#save the datasets
df04.to_pickle('df04')
df04a.to_pickle('df04a')
df59.to_pickle('df59')
df59a.to_pickle('df59a')




