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
import math
import os
import sklearn
import numpy as np
import json
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from src import GMM

plt.style.use('fivethirtyeight')


# ------------ HYPERPARAMETERS -------------
BASE_PATH = '../COVID-19/csse_covid_19_data/'
MIN_CASES = 1000
# ------------------------------------------

NUM_COLORS = 58
LINE_STYLES = ['solid', 'dashed', 'dotted']
NUM_STYLES = len(LINE_STYLES)

confirmed = os.path.join(
    BASE_PATH,
    'csse_covid_19_time_series',
    'time_series_covid19_confirmed_US.csv')
confirmed = data.load_csv_data(confirmed)
features = np.array([])
targets = np.array([])

# extract each state's case prograssion
for val in np.unique(confirmed["Province_State"]):
    df = data.filter_by_attribute(
        confirmed, "Province_State", val)
    cases, labels = data.get_cases_chronologically_US(df)
    cases = cases.sum(axis=0)
    if features.size == 0:
        features = cases
    elif val != 'Diamond Princess' and val != 'Grand Princess':
        features = np.vstack((features, cases))
    if val != 'Diamond Princess' and val != 'Grand Princess':
        targets = np.append(targets, labels[0])

print(features.shape)

# Smooth STATE progession curves and compute second derivative
# Idea to use a Savitsky-Golay filter comes from this stackoverflow page:
# https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way
sg_state_cases = np.zeros(features.shape)
for i in range(features.shape[0]):
    sg_state_cases[i,:] = savgol_filter(features[i,:], 51, 3) # window size 51, polynomial order 3

daily_cases = np.zeros(features.shape)
for i in range(1,features.shape[1]):
    daily_cases[:,i] = sg_state_cases[:,i] - sg_state_cases[:,i-1]

sg_daily_cases = np.zeros(features.shape)
for i in range(features.shape[0]):
    sg_daily_cases[i,:] = savgol_filter(daily_cases[i,:], 51, 3) # window size 51, polynomial order 3

deriv1 = np.zeros(features.shape)
for i in range(1,features.shape[1]):
    deriv1[:,i] = sg_daily_cases[:,i] - sg_daily_cases[:,i-1]

sg_deriv1= np.zeros(features.shape)
for i in range(features.shape[0]):
    sg_deriv1[i,:] = savgol_filter(deriv1[i,:], 51, 3) # window size 51, polynomial order 3

#########################################################################

# extract each county's case progression
county_cases = np.array([])
for val in np.unique(confirmed["FIPS"]):
    df = data.filter_by_attribute(
        confirmed, "FIPS", val)
    cases, labels = data.get_cases_chronologically_US(df)
    #print(cases)
    #cases = cases.sum(axis=0)
    if county_cases.size == 0:
        county_cases = cases
    elif not math.isnan(val):
        #print(val)
        county_cases = np.vstack((county_cases, cases))
print(county_cases.shape)

# Smooth data and compute second derivative of county data
# Idea to use a Savitsky-Golay filter comes from this stackoverflow page:
# https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way
sg_county_cases = np.zeros(county_cases.shape)
for i in range(county_cases.shape[0]):
    sg_county_cases[i,:] = savgol_filter(county_cases[i,:], 51, 3) # window size 51, polynomial order 3

daily_county_cases = np.zeros(county_cases.shape)
for i in range(1,county_cases.shape[1]):
    daily_county_cases[:,i] = sg_county_cases[:,i] - sg_county_cases[:,i-1]

sg_daily_county_cases = np.zeros(county_cases.shape)
for i in range(county_cases.shape[0]):
    sg_daily_county_cases[i,:] = savgol_filter(daily_county_cases[i,:], 51, 3) # window size 51, polynomial order 3

county_deriv1 = np.zeros(county_cases.shape)
for i in range(1,county_cases.shape[1]):
    county_deriv1[:,i] = sg_daily_county_cases[:,i] - sg_daily_county_cases[:,i-1]

sg_county_deriv1= np.zeros(county_cases.shape)
for i in range(county_cases.shape[0]):
    sg_county_deriv1[i,:] = savgol_filter(county_deriv1[i,:], 51, 3) # window size 51, polynomial order 3

# Compute negative to positive inflection points in state case progression data
max = np.zeros((5,2))
data1 = np.array([])
for state in range(features.shape[0]):
    count = 0
    nonzeros = np.nonzero(sg_deriv1[state,:])[0]
    ip_list = []
    if nonzeros.size != 0:
        #print(nonzeros)
        start_ind = nonzeros[0]
        start_sign = np.sign(sg_deriv1[state, start_ind])
        goal_sign = start_sign * -1
        #print(start_ind)
        for day in range(start_ind, features.shape[1]):
            if np.sign(sg_deriv1[state, day]) == goal_sign:
                start_sign = goal_sign
                goal_sign = start_sign * -1
                if start_sign == 1:
                    ip_list.append(day)
                    count += 1
    max[count, 0] = count
    max[count, 1] += 1
    while len(ip_list) < 4:
        ip_list.append(0)
    if state == 0:
        data1 = np.array(ip_list)
    else:
        data1 = np.vstack((data1, np.array(ip_list)))
#print(max)
for col in range(data1.shape[1]):
    avg = np.sum(data1[:,col])/np.nonzero(data1[:,col])[0].shape
    for state in range(data1.shape[0]):
        if data1[state,col] == 0:
            data1[state,col] = avg

# Compute positive to negative inflection points in state case progression data
max = np.zeros((6,2))
data2 = np.array([])
for state in range(features.shape[0]):
    count = 0
    nonzeros = np.nonzero(sg_deriv1[state,:])[0]
    ip_list = []
    if nonzeros.size != 0:
        #print(nonzeros)
        start_ind = nonzeros[0]
        start_sign = np.sign(sg_deriv1[state, start_ind])
        goal_sign = start_sign * -1
        #print(start_ind)
        for day in range(start_ind, features.shape[1]):
            if np.sign(sg_deriv1[state, day]) == goal_sign:
                start_sign = goal_sign
                goal_sign = start_sign * -1
                if start_sign == -1:
                    ip_list.append(day)
                    count += 1
    max[count, 0] = count
    max[count, 1] += 1
    while len(ip_list) < 4:
        ip_list.append(0)
    if state == 0:
        data2 = np.array(ip_list)
    else:
        data2 = np.vstack((data2, np.array(ip_list)))
#print(max)
for col in range(data2.shape[1]):
    avg = np.sum(data2[:,col])/np.nonzero(data2[:,col])[0].shape
    for state in range(data2.shape[0]):
        if data2[state,col] == 0:
            data2[state,col] = avg


data = np.concatenate((data1, data2), axis=1)
#print(data1.shape)
#print(data2.shape)
#print(data.shape)

# Compute negative to positive inflection points in COUNTY case progression data
county_data1 = np.array([])
max = np.zeros((14,2))
for county in range(county_cases.shape[0]):
    count = 0
    nonzeros = np.nonzero(sg_county_deriv1[county,:])[0]
    #print(sg_county_deriv1[county,:])
    ip_list = []
    if nonzeros.size != 0:
        #print(nonzeros)
        start_ind = nonzeros[0]
        start_sign = np.sign(sg_county_deriv1[county, start_ind])
        goal_sign = start_sign * -1
        #print(start_ind)
        for day in range(start_ind, county_cases.shape[1]):
            if np.sign(sg_county_deriv1[county, day]) == goal_sign:
                start_sign = goal_sign
                goal_sign = start_sign * -1
                if start_sign == 1:
                    ip_list.append(day)
                    count += 1
    max[count, 0] = count
    max[count, 1] += 1
    if len(ip_list) >= 1 and len(ip_list) <= 4:
        while len(ip_list) < 4:
            ip_list.append(0)
        if county_data1.size == 0:
            county_data1 = np.array(ip_list)
        else:
            county_data1 = np.vstack((county_data1, np.array(ip_list)))
print(max)
for col in range(county_data1.shape[1]):
    avg = np.sum(county_data1[:,col])/np.nonzero(county_data1[:,col])[0].shape
    for county in range(county_data1.shape[0]):
        if county_data1[county,col] == 0:
            county_data1[county,col] = avg

# Compute positive to negative inflection points in COUNTY case progression data
county_data2 = np.array([])
max = np.zeros((14,2))
for county in range(county_cases.shape[0]):
    count = 0
    nonzeros = np.nonzero(sg_county_deriv1[county,:])[0]
    #print(sg_county_deriv1[county,:])
    ip_list = []
    if nonzeros.size != 0:
        #print(nonzeros)
        start_ind = nonzeros[0]
        start_sign = np.sign(sg_county_deriv1[county, start_ind])
        goal_sign = start_sign * -1
        #print(start_ind)
        for day in range(start_ind, county_cases.shape[1]):
            if np.sign(sg_county_deriv1[county, day]) == goal_sign:
                start_sign = goal_sign
                goal_sign = start_sign * -1
                if start_sign == -1:
                    ip_list.append(day)
                    count += 1
    max[count, 0] = count
    max[count, 1] += 1
    if len(ip_list) >= 1 and len(ip_list) <= 4:
        while len(ip_list) < 4:
            ip_list.append(0)
        if county_data2.size == 0:
            county_data2 = np.array(ip_list)
        else:
            county_data2 = np.vstack((county_data2, np.array(ip_list)))
print(max)
for col in range(county_data2.shape[1]):
    avg = np.sum(county_data2[:,col])/np.nonzero(county_data2[:,col])[0].shape
    for county in range(county_data2.shape[0]):
        if county_data2[county,col] == 0:
            county_data2[county,col] = avg
county_data = np.concatenate((county_data1, county_data2[0:county_data1.shape[0],:]), axis=1)
print(county_data1.shape)
print(county_data2.shape)
print(county_data.shape)


gmm_learner = GMM(2, 'diagonal')
gmm_learner.fit(county_data)
labels = gmm_learner.predict(data)
cluster4 = features[labels == 4,:]
cluster3 = features[labels == 3,:]
cluster2 = features[labels == 2,:]
cluster1 = features[labels == 1,:]
cluster0 = features[labels == 0,:]
cluster4_targets = targets[labels == 4]
cluster3_targets = targets[labels == 3]
cluster2_targets = targets[labels == 2]
cluster1_targets = targets[labels == 1]
cluster0_targets = targets[labels == 0]

cm = plt.get_cmap('jet')
###
NUM_COLORS = cluster0.shape[0]
colors = [cm(i) for i in np.linspace(0, 1, NUM_COLORS)]
legend = []
handles = []
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111)
for i in range(cluster0.shape[0]):
    lines = ax.plot(cluster0[i,:], label=cluster0_targets[0])
    handles.append(lines[0])
    lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
    lines[0].set_color(colors[i])
    legend.append(cluster0_targets[i])

ax.set_ylabel('# of confirmed cases')
ax.set_xlabel("Time (days since Jan 22, 2020)")

plt.legend(handles, legend, bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4)
plt.tight_layout()
plt.savefig('results5/exp1 - cluster 0.png')
###

###
NUM_COLORS = cluster1.shape[0]
colors = [cm(i) for i in np.linspace(0, 1, NUM_COLORS)]
legend = []
handles = []
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111)
for i in range(cluster1.shape[0]):
    lines = ax.plot(cluster1[i,:], label=cluster1_targets[0])
    handles.append(lines[0])
    lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
    lines[0].set_color(colors[i])
    legend.append(cluster1_targets[i])

ax.set_ylabel('# of confirmed cases')
ax.set_xlabel("Time (days since Jan 22, 2020)")

plt.legend(handles, legend, bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4)
plt.tight_layout()
plt.savefig('results5/exp1 - cluster 1.png')
###
"""
###
NUM_COLORS = cluster2.shape[0]
colors = [cm(i) for i in np.linspace(0, 1, NUM_COLORS)]
legend = []
handles = []
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111)
for i in range(cluster2.shape[0]):
    lines = ax.plot(cluster2[i,:], label=cluster2_targets[0])
    handles.append(lines[0])
    lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
    lines[0].set_color(colors[i])
    legend.append(cluster2_targets[i])

ax.set_ylabel('# of confirmed cases')
ax.set_xlabel("Time (days since Jan 22, 2020)")

plt.legend(handles, legend, bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4)
plt.tight_layout()
plt.savefig('results6/exp2 - cluster 2.png')
###

###
NUM_COLORS = cluster3.shape[0]
colors = [cm(i) for i in np.linspace(0, 1, NUM_COLORS)]
legend = []
handles = []
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111)
for i in range(cluster3.shape[0]):
    lines = ax.plot(cluster3[i,:], label=cluster3_targets[0])
    handles.append(lines[0])
    lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
    lines[0].set_color(colors[i])
    legend.append(cluster3_targets[i])

ax.set_ylabel('# of confirmed cases')
ax.set_xlabel("Time (days since Jan 22, 2020)")

plt.legend(handles, legend, bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4)
plt.tight_layout()
plt.savefig('results5/exp3 - cluster 3.png')
###

###
NUM_COLORS = cluster4.shape[0]
colors = [cm(i) for i in np.linspace(0, 1, NUM_COLORS)]
legend = []
handles = []
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111)
for i in range(cluster4.shape[0]):
    lines = ax.plot(cluster4[i,:], label=cluster4_targets[0])
    handles.append(lines[0])
    lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
    lines[0].set_color(colors[i])
    legend.append(cluster4_targets[i])

ax.set_ylabel('# of confirmed cases')
ax.set_xlabel("Time (days since Jan 22, 2020)")

plt.legend(handles, legend, bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4)
plt.tight_layout()
plt.savefig('results6/exp4 - cluster 4.png')
###
"""

"""

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111)
cm = plt.get_cmap('jet')
for i in range(features.shape[0]):
    plt.plot(range(features.shape[1]), features[i,:])
plt.legend(targets, bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4)
plt.tight_layout()
plt.savefig('exp - case progression')

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111)
cm = plt.get_cmap('jet')
for i in range(cluster0.shape[0]):
    plt.plot(range(features.shape[1]), cluster0[i,:])
plt.legend(targets[labels == 0], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4)
plt.tight_layout()
plt.savefig('exp1 - cluster 0')
"""

##############################################################
"""
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
"""
