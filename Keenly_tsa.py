# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 20:07:13 2020

@author: Jianyong
"""

import sys
sys.path.append('C:\\Users\\Jianyong\\Anaconda3\\Lib\\site-packages')

import statsmodels
#show location of statsmodel: statsmodels.__file__

import os
path="D:\Dropbox\OMSA\ISYEPractice\Keenly\Data"
os.chdir(path)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date

# import the datasets
df04a=pd.read_pickle('df04a')
df59a=pd.read_pickle('df59a')
df09a=pd.concat([df04a, df59a])
del df04a
del df59a

# Clean data
df09b=df09a.loc[(df09a['StatusCode']==0) &(df09a['SignalQuality']>0)]

# Change into datetime format
df09b['Date']=pd.to_datetime(df09b['Timestamp'])
df09b['Year']=df09b['Date'].dt.year
df09b['Month']=df09b['Date'].dt.month
df09b['Day']=df09b['Date'].dt.day
df09b['Hour']=df09b['Date'].dt.hour
df09b['Minute']=df09b['Date'].dt.minute
df09b['Second']=df09b['Date'].dt.second
df09b['Date1']=pd.to_datetime(df09b[['Year','Month','Day','Hour','Minute','Second']])
df09b.sort_values(['Date1'],inplace=True )
df09b.set_index('Date1', inplace=True)
df09b.to_pickle('df09b')


PID=df09a['PatientID'].unique()

df09a1=df09b.drop_duplicates()
df09a1.drop(['Timestamp', 'StatusCode','DeviceID'], axis=1, inplace=True)

# get individual patient data
p1=df09a1.loc[(df09a1['PatientID']==5708323928145920)]
p2=df09a1.loc[(df09a1['PatientID']==5689413791121408)]
p3=df09a1.loc[(df09a1['PatientID']==5649050225344512)]
p4=df09a1.loc[(df09a1['PatientID']==5634263223369728)]
p5=df09a1.loc[(df09a1['PatientID']==5634999273390080)]
p6=df09a1.loc[(df09a1['PatientID']==5664378560970752)]
p7=df09a1.loc[(df09a1['PatientID']==5717994718101504)]
p8=df09a1.loc[(df09a1['PatientID']==5640060892348416)]
p9=df09a1.loc[(df09a1['PatientID']==5668600916475904)]

p10=df09a1.loc[(df09a1['PatientID']==5670794235478016)]
p11=df09a1.loc[(df09a1['PatientID']==5674053578784768)]
p12=df09a1.loc[(df09a1['PatientID']==5709989402378240)]
p13=df09a1.loc[(df09a1['PatientID']==5715233054130176)]
p14=df09a1.loc[(df09a1['PatientID']==5721589337292800)]
p15=df09a1.loc[(df09a1['PatientID']==5740379684995072)]
p16=df09a1.loc[(df09a1['PatientID']==5741031244955648)]
p17=df09a1.loc[(df09a1['PatientID']==5746055551385600)]
p18=df09a1.loc[(df09a1['PatientID']==5749328048029696)]

p19=df09a1.loc[(df09a1['PatientID']==5750197846016000)]
p20=df09a1.loc[(df09a1['PatientID']==5756989665705984)]
p21=df09a1.loc[(df09a1['PatientID']==5661232933634048)]
p22=df09a1.loc[(df09a1['PatientID']==5670864666230784)]
p23=df09a1.loc[(df09a1['PatientID']==5655638436741120)]
p24=df09a1.loc[(df09a1['PatientID']==5702797177651200)]
p25=df09a1.loc[(df09a1['PatientID']==5680680545550336)]
p26=df09a1.loc[(df09a1['PatientID']==5736697924943872)]
p27=df09a1.loc[(df09a1['PatientID']==5678164130922496)]

# Aggregate the data by hour for each patient
p1h=p1.groupby(['Year','Month','Day','Hour'], as_index=False).mean()
p1h['Date']=pd.to_datetime(p1h[['Year','Month','Day','Hour']])
p1h.set_index('Date', inplace=True)
ax=p1h['RespirationRate'].plot(title='Patient ID:5708323928145920' )
ax.set_ylabel('Mean respiration rate')


p2h=p2.groupby(['Year','Month','Day','Hour'], as_index=False).mean()
p2h['Date']=pd.to_datetime(p2h[['Year','Month','Day','Hour']])
p2h.set_index('Date', inplace=True)
ax=p2h['RespirationRate'].plot(title='Patient ID:5689413791121408' )
ax.set_ylabel('Mean respiration rate')

p3h=p3.groupby(['Year','Month','Day','Hour'], as_index=False).mean()
p3h['Date']=pd.to_datetime(p3h[['Year','Month','Day','Hour']])
p3h.set_index('Date', inplace=True)
ax=p3h['RespirationRate'].plot(title='Patient ID:5649050225344512' )
ax.set_ylabel('Mean respiration rate')


p4h=p4.groupby(['Year','Month','Day','Hour'], as_index=False).mean()
p4h['Date']=pd.to_datetime(p4h[['Year','Month','Day','Hour']])
p4h.set_index('Date', inplace=True)
ax=p4h['RespirationRate'].plot(title='Patient ID:5634263223369728' )
ax.set_ylabel('Mean respiration rate')

p5h=p5.groupby(['Year','Month','Day','Hour'], as_index=False).mean()
p5h['Date']=pd.to_datetime(p5h[['Year','Month','Day','Hour']])
p5h.set_index('Date', inplace=True)
ax=p5h['RespirationRate'].plot(title='Patient ID:5634999273390080' )
ax.set_ylabel('Mean respiration rate')


p6h=p6.groupby(['Year','Month','Day','Hour'], as_index=False).mean()
p6h['Date']=pd.to_datetime(p6h[['Year','Month','Day','Hour']])
p6h.set_index('Date', inplace=True)
ax=p6h['RespirationRate'].plot(title='Patient ID:5664378560970752' )
ax.set_ylabel('Mean respiration rate')

p7h=p7.groupby(['Year','Month','Day','Hour'], as_index=False).mean()
p7h['Date']=pd.to_datetime(p7h[['Year','Month','Day','Hour']])
p7h.set_index('Date', inplace=True)
ax=p7h['RespirationRate'].plot(title='Patient ID:5717994718101504' )
ax.set_ylabel('Mean respiration rate')

p8h=p8.groupby(['Year','Month','Day','Hour'], as_index=False).mean()
p8h['Date']=pd.to_datetime(p8h[['Year','Month','Day','Hour']])
p8h.set_index('Date', inplace=True)
ax=p8h['RespirationRate'].plot(title='Patient ID:5640060892348416' )
ax.set_ylabel('Mean respiration rate')

p9h=p9.groupby(['Year','Month','Day','Hour'], as_index=False).mean()
p9h['Date']=pd.to_datetime(p9h[['Year','Month','Day','Hour']])
p9h.set_index('Date', inplace=True)
ax=p9h['RespirationRate'].plot(title='Patient ID:5668600916475904' )
ax.set_ylabel('Mean respiration rate')

p10h=p10.groupby(['Year','Month','Day','Hour'], as_index=False).mean()
p10h['Date']=pd.to_datetime(p10h[['Year','Month','Day','Hour']])
p10h.set_index('Date', inplace=True)
ax=p10h['RespirationRate'].plot(title='Patient ID:5670794235478016' )
ax.set_ylabel('Mean respiration rate')

p11h=p11.groupby(['Year','Month','Day','Hour'], as_index=False).mean()
p11h['Date']=pd.to_datetime(p11h[['Year','Month','Day','Hour']])
p11h.set_index('Date', inplace=True)
ax=p11h['RespirationRate'].plot(title='Patient ID:5674053578784768' )
ax.set_ylabel('Mean respiration rate')

p12h=p12.groupby(['Year','Month','Day','Hour'], as_index=False).mean()
p12h['Date']=pd.to_datetime(p12h[['Year','Month','Day','Hour']])
p12h.set_index('Date', inplace=True)
ax=p12h['RespirationRate'].plot(title='Patient ID:5709989402378240' )
ax.set_ylabel('Mean respiration rate')

p13h=p13.groupby(['Year','Month','Day','Hour'], as_index=False).mean()
p13h['Date']=pd.to_datetime(p13h[['Year','Month','Day','Hour']])
p13h.set_index('Date', inplace=True)
ax=p13h['RespirationRate'].plot(title='Patient ID:5715233054130176' )
ax.set_ylabel('Mean respiration rate')

p14h=p14.groupby(['Year','Month','Day','Hour'], as_index=False).mean()
p14h['Date']=pd.to_datetime(p14h[['Year','Month','Day','Hour']])
p14h.set_index('Date', inplace=True)
ax=p14h['RespirationRate'].plot(title='Patient ID:5721589337292800' )
ax.set_ylabel('Mean respiration rate')

p15h=p15.groupby(['Year','Month','Day','Hour'], as_index=False).mean()
p15h['Date']=pd.to_datetime(p15h[['Year','Month','Day','Hour']])
p15h.set_index('Date', inplace=True)
ax=p15h['RespirationRate'].plot(title='Patient ID:5740379684995072' )
ax.set_ylabel('Mean respiration rate')

p16h=p16.groupby(['Year','Month','Day','Hour'], as_index=False).mean()
p16h['Date']=pd.to_datetime(p16h[['Year','Month','Day','Hour']])
p16h.set_index('Date', inplace=True)
ax=p16h['RespirationRate'].plot(title='Patient ID:5741031244955648' )
ax.set_ylabel('Mean respiration rate')


p17h=p17.groupby(['Year','Month','Day','Hour'], as_index=False).mean()
p17h['Date']=pd.to_datetime(p17h[['Year','Month','Day','Hour']])
p17h.set_index('Date', inplace=True)
ax=p17h['RespirationRate'].plot(title='Patient ID:5746055551385600' )
ax.set_ylabel('Mean respiration rate')

'''
p18h=p18.groupby(['Year','Month','Day','Hour'], as_index=False).mean()
p18h['Date']=pd.to_datetime(p18h[['Year','Month','Day','Hour']])
p18h.set_index('Date', inplace=True)
ax=p18h['RespirationRate'].plot(title='Patient ID:5749328048029696' )
ax.set_ylabel('Mean respiration rate')
'''
p19h=p19.groupby(['Year','Month','Day','Hour'], as_index=False).mean()
p19h['Date']=pd.to_datetime(p19h[['Year','Month','Day','Hour']])
p19h.set_index('Date', inplace=True)
ax=p19h['RespirationRate'].plot(title='Patient ID:5750197846016000' )
ax.set_ylabel('Mean respiration rate')


p20h=p20.groupby(['Year','Month','Day','Hour'], as_index=False).mean()
p20h['Date']=pd.to_datetime(p20h[['Year','Month','Day','Hour']])
p20h.set_index('Date', inplace=True)
ax=p20h['RespirationRate'].plot(title='Patient ID:5756989665705984' )
ax.set_ylabel('Mean respiration rate')

p21h=p21.groupby(['Year','Month','Day','Hour'], as_index=False).mean()
p21h['Date']=pd.to_datetime(p21h[['Year','Month','Day','Hour']])
p21h.set_index('Date', inplace=True)
ax=p21h['RespirationRate'].plot(title='Patient ID:5661232933634048' )
ax.set_ylabel('Mean respiration rate')

p22h=p22.groupby(['Year','Month','Day','Hour'], as_index=False).mean()
p22h['Date']=pd.to_datetime(p22h[['Year','Month','Day','Hour']])
p22h.set_index('Date', inplace=True)
ax=p22h['RespirationRate'].plot(title='Patient ID:5670864666230784' )
ax.set_ylabel('Mean respiration rate')

p23h=p23.groupby(['Year','Month','Day','Hour'], as_index=False).mean()
p23h['Date']=pd.to_datetime(p23h[['Year','Month','Day','Hour']])
p23h.set_index('Date', inplace=True)
ax=p23h['RespirationRate'].plot(title='Patient ID:5655638436741120' )
ax.set_ylabel('Mean respiration rate')

p24h=p24.groupby(['Year','Month','Day','Hour'], as_index=False).mean()
p24h['Date']=pd.to_datetime(p24h[['Year','Month','Day','Hour']])
p24h.set_index('Date', inplace=True)
ax=p24h['RespirationRate'].plot(title='Patient ID:5702797177651200' )
ax.set_ylabel('Mean respiration rate')

p25h=p25.groupby(['Year','Month','Day','Hour'], as_index=False).mean()
p25h['Date']=pd.to_datetime(p25h[['Year','Month','Day','Hour']])
p25h.set_index('Date', inplace=True)
ax=p25h['RespirationRate'].plot(title='Patient ID:5680680545550336' )
ax.set_ylabel('Mean respiration rate')


p26h=p26.groupby(['Year','Month','Day','Hour'], as_index=False).mean()
p26h['Date']=pd.to_datetime(p26h[['Year','Month','Day','Hour']])
p26h.set_index('Date', inplace=True)
ax=p26h['RespirationRate'].plot(title='Patient ID:5736697924943872' )
ax.set_ylabel('Mean respiration rate')

p27h=p27.groupby(['Year','Month','Day','Hour'], as_index=False).mean()
p27h['Date']=pd.to_datetime(p27h[['Year','Month','Day','Hour']])
p27h.set_index('Date', inplace=True)
ax=p27h['RespirationRate'].plot(title='Patient ID:5678164130922496' )
ax.set_ylabel('Mean respiration rate')

p1h['PID']=1 
p2h['PID']=2
p3h['PID']=3
p4h['PID']=4
p5h['PID']=5 
p6h['PID']=6
p7h['PID']=7 
p8h['PID']=8 
p9h['PID']=9

p10h['PID']=10 
p11h['PID']=11 
p12h['PID']=12
p13h['PID']=13
p14h['PID']=14 
p15h['PID']=15
p16h['PID']=16
p17h['PID']=17 
#p18h['PID']=18

p19h['PID']=19 
p20h['PID']=20 
p21h['PID']=21
p22h['PID']=22 
p23h['PID']=23 
p24h['PID']=24
p25h['PID']=25 
p26h['PID']=26 
p27h['PID']=27


#####################################################################################

########## Seasona-trend decoposation  ##############################################

#####################################################################################
from statsmodels.tsa.seasonal import seasonal_decompose

p1d = seasonal_decompose(p1h['RespirationRate'], model='additive', freq=12)
p1d.plot()
p1h['Resid']=p1d.resid.fillna(0) #residual of the ST decomposition

p2d = seasonal_decompose(p2h['RespirationRate'], model='additive', freq=12)
p2d.plot()
p2h['Resid']=p2d.resid.fillna(0) 

p3d = seasonal_decompose(p3h['RespirationRate'], model='additive', freq=12)
p3d.plot()
p3h['Resid']=p3d.resid.fillna(0) 

p4d = seasonal_decompose(p4h['RespirationRate'], model='additive', freq=12)
p4d.plot()
p4h['Resid']=p4d.resid.fillna(0) 

p5d = seasonal_decompose(p5h['RespirationRate'], model='additive', freq=12)
p5d.plot()
p5h['Resid']=p5d.resid.fillna(0) 

p6d = seasonal_decompose(p6h['RespirationRate'], model='additive', freq=12)
p6d.plot()
p6h['Resid']=p6d.resid.fillna(0) 

p7d = seasonal_decompose(p7h['RespirationRate'], model='additive', freq=12)
p7d.plot()
p7h['Resid']=p7d.resid.fillna(0) 

p8d = seasonal_decompose(p8h['RespirationRate'], model='additive', freq=12)
p8d.plot()
p8h['Resid']=p8d.resid.fillna(0) 

p9d = seasonal_decompose(p9h['RespirationRate'], model='additive', freq=12)
p9d.plot()
p9h['Resid']=p9d.resid.fillna(0) 

p10d = seasonal_decompose(p10h['RespirationRate'], model='additive', freq=12)
p10d.plot()
p10h['Resid']=p10d.resid.fillna(0) 

p11d = seasonal_decompose(p11h['RespirationRate'], model='additive', freq=12)
p11d.plot()
p11h['Resid']=p11d.resid.fillna(0) 

p12d = seasonal_decompose(p12h['RespirationRate'], model='additive', freq=12)
p12d.plot()
p12h['Resid']=p12d.resid.fillna(0) 

p13d = seasonal_decompose(p13h['RespirationRate'], model='additive', freq=12)
p13d.plot()
p13h['Resid']=p13d.resid.fillna(0) 

p14d = seasonal_decompose(p14h['RespirationRate'], model='additive', freq=12)
p14d.plot()
p14h['Resid']=p14d.resid.fillna(0) 

p15d = seasonal_decompose(p15h['RespirationRate'], model='additive', freq=12)
p15d.plot()
p15h['Resid']=p15d.resid.fillna(0) 

p16d = seasonal_decompose(p16h['RespirationRate'], model='additive', freq=12)
p16d.plot()
p16h['Resid']=p16d.resid.fillna(0) 

p17d = seasonal_decompose(p17h['RespirationRate'], model='additive', freq=12)
p17d.plot()
p17h['Resid']=p17d.resid.fillna(0) 

'''
p18d = seasonal_decompose(p18h['RespirationRate'], model='additive', freq=12)
p18d.plot()
p18h['Resid']=p18d.resid.fillna(0) 
'''

p19d = seasonal_decompose(p19h['RespirationRate'], model='additive', freq=12)
p19d.plot()
p19h['Resid']=p19d.resid.fillna(0) 

p20d = seasonal_decompose(p20h['RespirationRate'], model='additive', freq=12)
p20d.plot()
p20h['Resid']=p20d.resid.fillna(0) 

p21d = seasonal_decompose(p21h['RespirationRate'], model='additive', freq=12)
p21d.plot()
p21h['Resid']=p21d.resid.fillna(0) 

p22d = seasonal_decompose(p22h['RespirationRate'], model='additive', freq=12)
p22d.plot()
p22h['Resid']=p22d.resid.fillna(0) 

p23d = seasonal_decompose(p23h['RespirationRate'], model='additive', freq=12)
p23d.plot()
p23h['Resid']=p23d.resid.fillna(0) 

p24d = seasonal_decompose(p24h['RespirationRate'], model='additive', freq=12)
p24d.plot()
p24h['Resid']=p24d.resid.fillna(0) 

p25d = seasonal_decompose(p25h['RespirationRate'], model='additive', freq=12)
p25d.plot()
p25h['Resid']=p25d.resid.fillna(0) 

p26d = seasonal_decompose(p26h['RespirationRate'], model='additive', freq=12)
p26d.plot()
p26h['Resid']=p26d.resid.fillna(0) 

p27d = seasonal_decompose(p27h['RespirationRate'], model='additive', freq=12)
p27d.plot()
p27h['Resid']=p27d.resid.fillna(0) 

######## detect anormaly using generalized ESD test

from sesd import generalized_esd

a1d=generalized_esd(p1h['Resid'], max_anomalies=16, alpha=0.05, hybrid=False)
a1=[0]*len(p1h)
for i in a1d: a1[i]=1
p1h['AnormalyRes']=a1  #1: yes, 0: no
p1h1=p1h.copy()
p1h1.loc[p1h1['AnormalyRes'] == 0,'Resid'] = np.nan
p1h['Resid'].plot(label='Residual')
plt.plot(p1h1.index, p1h1['Resid'],  marker='o', markersize=3, color='red', label='Anomaly')
plt.legend()
plt.ylabel('Residual of ST decomposition')

a2d=generalized_esd(p2h['Resid'], max_anomalies=16, alpha=0.05, hybrid=False)
a2=[0]*len(p2h)
for i in a2d: a2[i]=1
p2h['AnormalyRes']=a2  #1: yes, 0: no

a3d=generalized_esd(p3h['Resid'], max_anomalies=16, alpha=0.05, hybrid=False)
a3=[0]*len(p3h)
for i in a3d: a3[i]=1
p3h['AnormalyRes']=a3  #1: yes, 0: no

a4d=generalized_esd(p4h['Resid'], max_anomalies=16, alpha=0.05, hybrid=False)
a4=[0]*len(p4h)
for i in a4d: a4[i]=1
p4h['AnormalyRes']=a4  #1: yes, 0: no


a5d=generalized_esd(p5h['Resid'], max_anomalies=16, alpha=0.05, hybrid=False)
a5=[0]*len(p5h)
for i in a5d: a5[i]=1
p5h['AnormalyRes']=a5  #1: yes, 0: no

a6d=generalized_esd(p6h['Resid'], max_anomalies=16, alpha=0.05, hybrid=False)
a6=[0]*len(p6h)
for i in a6d: a6[i]=1
p6h['AnormalyRes']=a6  #1: yes, 0: no

a7d=generalized_esd(p7h['Resid'], max_anomalies=16, alpha=0.05, hybrid=False)
a7=[0]*len(p7h)
for i in a7d: a7[i]=1
p7h['AnormalyRes']=a7  #1: yes, 0: no

a8d=generalized_esd(p8h['Resid'], max_anomalies=16, alpha=0.05, hybrid=False)
a8=[0]*len(p8h)
for i in a8d: a8[i]=1
p8h['AnormalyRes']=a8  #1: yes, 0: no

a9d=generalized_esd(p9h['Resid'], max_anomalies=16, alpha=0.05, hybrid=False)
a9=[0]*len(p9h)
for i in a9d: a9[i]=1
p9h['AnormalyRes']=a9  #1: yes, 0: no

a10d=generalized_esd(p10h['Resid'], max_anomalies=16, alpha=0.05, hybrid=False)
a10=[0]*len(p10h)
for i in a10d: a10[i]=1
p10h['AnormalyRes']=a10  #1: yes, 0: no

a11d=generalized_esd(p11h['Resid'], max_anomalies=16, alpha=0.05, hybrid=False)
a11=[0]*len(p11h)
for i in a11d: a11[i]=1
p11h['AnormalyRes']=a11  #1: yes, 0: no

a12d=generalized_esd(p12h['Resid'], max_anomalies=16, alpha=0.05, hybrid=False)
a12=[0]*len(p12h)
for i in a12d: a12[i]=1
p12h['AnormalyRes']=a12  #1: yes, 0: no

a13d=generalized_esd(p13h['Resid'], max_anomalies=16, alpha=0.05, hybrid=False)
a13=[0]*len(p13h)
for i in a13d: a13[i]=1
p13h['AnormalyRes']=a13  #1: yes, 0: no

a14d=generalized_esd(p14h['Resid'], max_anomalies=16, alpha=0.05, hybrid=False)
a14=[0]*len(p14h)
for i in a14d: a14[i]=1
p14h['AnormalyRes']=a14  #1: yes, 0: no

a15d=generalized_esd(p15h['Resid'], max_anomalies=16, alpha=0.05, hybrid=False)
a15=[0]*len(p15h)
for i in a15d: a15[i]=1
p15h['AnormalyRes']=a15  #1: yes, 0: no

a16d=generalized_esd(p16h['Resid'], max_anomalies=16, alpha=0.05, hybrid=False)
a16=[0]*len(p16h)
for i in a16d: a16[i]=1
p16h['AnormalyRes']=a16  #1: yes, 0: no

a17d=generalized_esd(p17h['Resid'], max_anomalies=16, alpha=0.05, hybrid=False)
a17=[0]*len(p17h)
for i in a17d: a17[i]=1
p17h['AnormalyRes']=a17  #1: yes, 0: no

'''
a18d=generalized_esd(p18h['Resid'], max_anomalies=16, alpha=0.05, hybrid=False)
a18=[1]*len(p18h)
for i in a18d: a18[i]=1
p18h['AnormalyRes']=a18  #1: yes, 0: no
'''

a19d=generalized_esd(p19h['Resid'], max_anomalies=16, alpha=0.05, hybrid=False)
a19=[0]*len(p19h)
for i in a19d: a19[i]=1
p19h['AnormalyRes']=a19  #1: yes, 0: no

a20d=generalized_esd(p20h['Resid'], max_anomalies=16, alpha=0.05, hybrid=False)
a20=[0]*len(p20h)
for i in a20d: a20[i]=1
p20h['AnormalyRes']=a20  #1: yes, 0: no

a21d=generalized_esd(p21h['Resid'], max_anomalies=16, alpha=0.05, hybrid=False)
a21=[0]*len(p21h)
for i in a21d: a21[i]=1
p21h['AnormalyRes']=a21  #1: yes, 0: no

a22d=generalized_esd(p22h['Resid'], max_anomalies=16, alpha=0.05, hybrid=False)
a22=[0]*len(p22h)
for i in a22d: a22[i]=1
p22h['AnormalyRes']=a22  #1: yes, 0: no

a23d=generalized_esd(p23h['Resid'], max_anomalies=16, alpha=0.05, hybrid=False)
a23=[0]*len(p23h)
for i in a23d: a23[i]=1
p23h['AnormalyRes']=a23  #1: yes, 0: no

a24d=generalized_esd(p24h['Resid'], max_anomalies=16, alpha=0.05, hybrid=False)
a24=[0]*len(p24h)
for i in a24d: a24[i]=1
p24h['AnormalyRes']=a24  #1: yes, 0: no

a25d=generalized_esd(p25h['Resid'], max_anomalies=16, alpha=0.05, hybrid=False)
a25=[0]*len(p25h)
for i in a25d: a25[i]=1
p25h['AnormalyRes']=a25  #1: yes, 0: no

a26d=generalized_esd(p26h['Resid'], max_anomalies=16, alpha=0.05, hybrid=False)
a26=[0]*len(p26h)
for i in a26d: a26[i]=1
p26h['AnormalyRes']=a26  #1: yes, 0: no

a27d=generalized_esd(p27h['Resid'], max_anomalies=16, alpha=0.05, hybrid=False)
a27=[0]*len(p27h)
for i in a27d: a27[i]=1
p27h['AnormalyRes']=a27  #1: yes, 0: no



# save the data;
p1_27h=pd.concat([p1h, p2h, p3h, p4h, p5h, p6h, p7h,p8h,p9h, p10h,p11h, p12h, p13h, p14h, p15h, p16h, p17h,p19h, p20h, p21h, p22h, p23h, p24h, p25h, p26h, p27h])

p1_27h.describe()

p1_27h.to_pickle('p1_27h')


# Get variance
p1v=p1.groupby(['Year','Month','Day','Hour'], as_index=False).var()
p2v=p2.groupby(['Year','Month','Day','Hour'], as_index=False).var()
p3v=p3.groupby(['Year','Month','Day','Hour'], as_index=False).var()
p4v=p4.groupby(['Year','Month','Day','Hour'], as_index=False).var()
p5v=p5.groupby(['Year','Month','Day','Hour'], as_index=False).var()
p6v=p6.groupby(['Year','Month','Day','Hour'], as_index=False).var()
p7v=p7.groupby(['Year','Month','Day','Hour'], as_index=False).var()
p8v=p8.groupby(['Year','Month','Day','Hour'], as_index=False).var()
p9v=p9.groupby(['Year','Month','Day','Hour'], as_index=False).var()


p10v=p10.groupby(['Year','Month','Day','Hour'], as_index=False).var()
p11v=p11.groupby(['Year','Month','Day','Hour'], as_index=False).var()
p12v=p12.groupby(['Year','Month','Day','Hour'], as_index=False).var()
p13v=p13.groupby(['Year','Month','Day','Hour'], as_index=False).var()
p14v=p14.groupby(['Year','Month','Day','Hour'], as_index=False).var()
p15v=p15.groupby(['Year','Month','Day','Hour'], as_index=False).var()
p16v=p16.groupby(['Year','Month','Day','Hour'], as_index=False).var()
p17v=p17.groupby(['Year','Month','Day','Hour'], as_index=False).var()
p18v=p18.groupby(['Year','Month','Day','Hour'], as_index=False).var()
p19v=p19.groupby(['Year','Month','Day','Hour'], as_index=False).var()


p20v=p20.groupby(['Year','Month','Day','Hour'], as_index=False).var()
p21v=p21.groupby(['Year','Month','Day','Hour'], as_index=False).var()
p22v=p22.groupby(['Year','Month','Day','Hour'], as_index=False).var()
p23v=p23.groupby(['Year','Month','Day','Hour'], as_index=False).var()
p24v=p24.groupby(['Year','Month','Day','Hour'], as_index=False).var()
p25v=p25.groupby(['Year','Month','Day','Hour'], as_index=False).var()
p26v=p26.groupby(['Year','Month','Day','Hour'], as_index=False).var()
p27v=p27.groupby(['Year','Month','Day','Hour'], as_index=False).var()

p1v['PID']=1
p2v['PID']=2
p3v['PID']=3
p4v['PID']=4
p5v['PID']=5
p6v['PID']=6
p7v['PID']=7
p8v['PID']=8
p9v['PID']=9

p10v['PID']=10
p11v['PID']=11
p12v['PID']=12
p13v['PID']=13
p14v['PID']=14
p15v['PID']=15
p16v['PID']=16
p17v['PID']=17
p18v['PID']=18
p19v['PID']=19

p20v['PID']=20
p21v['PID']=21
p22v['PID']=22
p23v['PID']=23
p24v['PID']=24
p25v['PID']=25
p26v['PID']=26
p27v['PID']=27



#save the data
p1_27v=pd.concat([p1v, p2v, p3v, p4v, p5v, p6v, p7v,p8v,p9v, p10v,p11v, p12v, p13v, p14v, p15v, p16v, p17v,p19v, p20v, p21v, p22v, p23v, p24v, p25v, p26v, p27v])

p1_27v.to_pickle('p1_27v')



















