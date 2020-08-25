"""
Experiment summary
------------------
Treat each province/state in a country cases over time
as a vector, do a simple K-Nearest Neighbor between
countries. What country has the most similar trajectory
to a given country?
"""

import sys
sys.path.insert(0, '..')

from utils import data
import os
import sklearn
import numpy as np
import json
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

plt.style.use('fivethirtyeight')


# ------------ HYPERPARAMETERS -------------
BASE_PATH = '../COVID-19/csse_covid_19_data/'
MIN_CASES = 1000
# ------------------------------------------

confirmed = os.path.join(
    BASE_PATH,
    'csse_covid_19_time_series',
    'time_series_covid19_confirmed_US.csv')
confirmed = data.load_csv_data(confirmed)
features = np.array([])
targets = np.array([])


for val in np.unique(confirmed["Province_State"]):
    df = data.filter_by_attribute(
        confirmed, "Province_State", val)
    cases, labels = data.get_cases_chronologically_US(df)
    cases = cases.sum(axis=0)
    if features.size == 0:
        features = cases
    else:
        features = np.vstack((features, cases))
    targets = np.append(targets, labels[0])

daily_cases = np.zeros(features.shape)
for i in range(1,features.shape[1]):
    daily_cases[:,i] = features[:,i] - features[:,i-1]

sg_daily_cases = np.zeros(features.shape)
for i in range(features.shape[0]):
    sg_daily_cases[i,:] = savgol_filter(daily_cases[i,:], 51, 3) # window size 51, polynomial order 3

case_rolling_avg = np.zeros(features.shape)
for i in range(14,features.shape[1]):
    case_rolling_avg[:,i] = np.average(daily_cases[:,i-14:i], axis=1)

sg_daily_cases_avged = np.zeros(features.shape)
for i in range(features.shape[0]):
    sg_daily_cases_avged[i,:] = savgol_filter(case_rolling_avg[i,:], 51, 3) # window size 51, polynomial order 3

deriv1 = np.zeros(features.shape)
for i in range(1,features.shape[1]):
    deriv1[:,i] = sg_daily_cases_avged[:,i] - sg_daily_cases_avged[:,i-1]

sg_deriv1= np.zeros(features.shape)
for i in range(features.shape[0]):
    sg_deriv1[i,:] = savgol_filter(deriv1[i,:], 51, 3) # window size 51, polynomial order 3

deriv2 = np.zeros(features.shape)
for i in range(1,features.shape[1]):
    deriv2[:,i] = sg_deriv1[:,i] - sg_deriv1[:,i-1]

sg_deriv2= np.zeros(features.shape)
for i in range(features.shape[0]):
    sg_deriv2[i,:] = savgol_filter(deriv2[i,:], 51, 3) # window size 51, polynomial order 3

data = np.array([])
for state in range(features.shape[0]):
    count = 0
    nonzeros = np.nonzero(deriv2[state,:])[0]
    ip_list = []
    if nonzeros.size != 0:
        #print(nonzeros)
        start_ind = nonzeros[0]
        start_sign = np.sign(deriv2[state, start_ind])
        goal_sign = start_sign * -1
        #print(start_ind)
        for day in range(start_ind, features.shape[1]):
            if np.sign(deriv2[state, day]) == goal_sign:
                start_sign = goal_sign
                goal_sign = start_sign * -1
                ip_list.append(day)
                count += 1
    while len(ip_list) < 14:
        ip_list.append(-1)
    if state == 0:
        data = np.array(ip_list)
    else:
        data = np.vstack((data, np.array(ip_list)))
print(data)


##############################################################

plt.figure()
for i in range(features.shape[0]):
    plt.plot(range(features.shape[1]), daily_cases[i,:])
plt.savefig('daily cases')

plt.figure()
for i in range(features.shape[0]):
    plt.plot(range(features.shape[1]), sg_daily_cases[i,:])
plt.savefig('savgol filtered daily cases')

plt.figure()
for i in range(features.shape[0]):
    plt.plot(range(features.shape[1]), sg_daily_cases_avged[i,:])
plt.savefig('savgol filtered daily cases rolling averaged')

plt.figure()
for i in range(features.shape[0]):
    plt.plot(range(features.shape[1]), case_rolling_avg[i,:])
plt.savefig('daily cases rolling average')

plt.figure()
for i in range(features.shape[0]):
    plt.plot(range(features.shape[1]), features[i,:])
plt.savefig('case progression')

plt.figure()
for i in range(features.shape[0]):
    plt.plot(range(features.shape[1]), sg_deriv1[i,:])
plt.savefig('deriv1')

plt.figure()
for i in range(features.shape[0]):
    plt.plot(range(features.shape[1]), sg_deriv2[i,:])
plt.savefig('deriv2')
