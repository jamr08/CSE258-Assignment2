#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gzip
from collections import defaultdict
import math
import scipy.optimize
import scipy
import numpy as np
import string
import random
import string
from sklearn import linear_model
import csv
import os
import statistics as st
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import pandas as pd


# In[2]:


out_path = "processed_endomondoHR_proper.npy"

data = np.load(out_path, allow_pickle= True)[0]


# In[3]:


# creating some dictionaries for data storage
user_gender = defaultdict() # user and their gender
sport_fem = defaultdict()
sport_male = defaultdict()
sport_uk = defaultdict()
user_sport = defaultdict(list) # user and all the sports they play
sport_user = defaultdict(list)


# In[4]:


user_sport_times = defaultdict(dict) # duration of a sport each time a user plays it
sport_times = defaultdict(list) # duration of each sport
user_times = defaultdict(list) # duration the person workouts for

user_sport_avghr = defaultdict(dict) # avg heartrate of a sport each time a user plays it
sport_avghr = defaultdict(list) # list of avghr of each user for the sport
user_avghr = defaultdict(list) # list of avghr of each user for all sport


sport_list = [] #list of all the sports


sport_start = defaultdict(list) # when is a sport/user active i.e. when is a sport started timestamp[0]
user_start = defaultdict(list)

sport_alt_change = defaultdict(list) # total change in altitude during a run of the sport for every run and sport
user_alt_change = defaultdict(list)

sport_speed = defaultdict(list) # average speeds of the sport and the user
user_speed = defaultdict(list)


# In[ ]:


# data[0]['altitude']
# mi = min(data[0]['altitude'])
# ma = max(data[0]['altitude'])
# diff = abs(ma - mi)
# print(diff)

#data[0]['tar_derived_speed']


# In[5]:


for d in data:
    user_gender[d['userId']] = d['gender'] # user and gender

    if d['gender'] == 'female': #sports per gender
        if d['sport'] not in sport_fem.keys():
            sport_fem[d['sport']] = 0
        sport_fem[d['sport']] += 1
    
    elif d['gender'] == 'male':
        if d['sport'] not in sport_male.keys():
            sport_male[d['sport']] = 0
        sport_male[d['sport']] += 1

    else:
        if d['sport'] not in sport_uk.keys():
            sport_uk[d['sport']] = 0
        sport_uk[d['sport']] += 1
    
    date = datetime.fromtimestamp(d['timestamp'][0]).strftime("%Y-%m-%d") # extracting the month in which a run of the sport was played
    user_start[d['userId']].append(datetime.strptime(date, "%Y-%m-%d").month)
    sport_start[d['sport']].append(datetime.strptime(date, "%Y-%m-%d").month)


    if d['sport'] not in sport_list: #all sports
        sport_list.append(d['sport'])

    if d['sport'] not in user_sport[d['userId']]: #all sports a user plays
        user_sport[d['userId']].append(d['sport'])
    
    if d['userId'] not in sport_user[d['sport']]: #users who play a sport
        sport_user[d['sport']].append(d['userId'])

    durtime = (d['timestamp'][-1] - d['timestamp'][0] ) / (3600) # duration in hours
    #user_sport_times[d['userId']][d['sport']].append(durtime)
    if d['userId'] not in user_sport_times.keys():
        user_sport_times[d['userId']] = {}
    if d['sport'] not in user_sport_times[d['userId']].keys():
        user_sport_times[d['userId']][d['sport']] = []
    user_sport_times[d['userId']][d['sport']].append(durtime) # user-sport and the time they played it

    sport_times[d['sport']].append(durtime)
    user_times[d['userId']].append(durtime)
    
    avgtarhr = np.mean(d['tar_heart_rate']) #avg tar hr
    #user_sport_avghr[d['userId']][d['sport']].append(avgtarhr)
    if d['userId'] not in user_sport_avghr.keys():
        user_sport_avghr[d['userId']] = {}
    if d['sport'] not in user_sport_avghr[d['userId']].keys():
        user_sport_avghr[d['userId']][d['sport']] = []
    user_sport_avghr[d['userId']][d['sport']].append(avgtarhr)
    sport_avghr[d['sport']].append(avgtarhr)
    user_avghr[d['userId']].append(avgtarhr)


    mi = min(d['altitude']) #altitude
    ma = max(d['altitude'])
    diff = abs(ma - mi)
    sport_alt_change[d['sport']].append(diff)
    user_alt_change[d['userId']].append(diff)

    avg_tar_speed = np.mean(d['tar_derived_speed']) #avg tar dervied speed
    sport_speed[d['sport']].append(avg_tar_speed)
    user_speed[d['userId']].append(avg_tar_speed)


# In[6]:


# create dataset for queries
query_data = defaultdict()

for u, g in user_gender.items():
    unplayed = []
    for sp in sport_list:
        if sp in user_sport[u]:
            query_data[(u,sp)] = 1
        
        else:
            unplayed.append(sp)
    
    for i in range (0, len(user_sport[u])): # create as many 0 entries for a person as 1
        sp = random.choice(unplayed)
        unplayed.remove(sp)
        query_data[(u,sp)] = 0

temp_query = list(query_data.items())
random.shuffle(temp_query)
query_data = dict(temp_query)
x_train, x_test, y_train, y_test = train_test_split(list(query_data.keys()), list(query_data.values()))


# In[8]:


#some user demo graphics
m, f, u = 0, 0, 0
for k,v in user_gender.items():
    if v == 'male':
        m += 1
    elif v == 'female':
        f += 1
    else:
        u += 1

categories = ['male', 'female', 'unknown']
print(m,f,u)
counts = [m, f, u]
plt.xlabel("Gender")
plt.ylabel("# of Players")
plt.title("User Demographics")
plt.bar(categories, counts, width=0.5)
plt.savefig('Demographics')


# In[9]:


#sport popularity in total 
sport_pop = {}
sport_user = dict(sorted(sport_user.items()))
categories = sport_user.keys()
counts = [len(u) for u in sport_user.values()]
sport_pop = {k:v/1059 for (k,v) in zip(categories,counts)}
plt.xlabel("Sport")
plt.ylabel("# of Players")
plt.title("Sport Popularity")
plt.bar(categories, counts, width=0.5)
plt.xticks(rotation=90)
plt.savefig('Sport Popularity', bbox_inches="tight")
print(len(user_sport))


# In[10]:


# ## code to turn sport into a one hot vector 

# %%
for sp in sport_list:
    if sp not in sport_male.keys():
        sport_male[sp] = 0

    if sp not in sport_fem.keys():
        sport_fem[sp] = 0
    
    if sp not in sport_uk.keys():
        sport_uk[sp] = 0


# %%
print(len(sport_male), len(sport_fem), len(sport_uk))


# In[11]:


sorted_counts = defaultdict(list) #male,female, uk
for sp in sport_list:
    sorted_counts[sp].append(sport_male[sp]) #
    sorted_counts[sp].append(sport_fem[sp])
    sorted_counts[sp].append(sport_uk[sp])


# In[12]:


sports_vals = sorted_counts.keys()
gender_vals = ['male', 'female', 'unknown']
counts = np.array(list(sorted_counts.values()))
fig, ax = plt.subplots(figsize=(20, 20))
im = ax.imshow(counts)

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(gender_vals)), labels=gender_vals)
ax.set_yticks(np.arange(len(sports_vals)), labels=sports_vals)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(sports_vals)):
    for j in range(len(gender_vals)):
        text = ax.text(j, i, counts[i, j], ha="center", va="center", color="w")

ax.set_title("Sport Popularity by Gender")
fig.tight_layout()
# plt.show()
plt.savefig('sport-gender-fraction')


# In[13]:


# sport intersections
sport_intersections = {}

for s1 in sport_user:
    for s2 in sport_user:
        if s1 == s2:
            # print(s1,s2,":", len(sport_user[s1]))
            sport_intersections[(s1,s2)] = len(sport_user[s1]) #0
            continue
        # if (s2,s1) in sport_intersections.keys():
        #     continue
        intersect = len(list(set(sport_user[s1]).intersection(sport_user[s2])))
        sport_intersections[(s1,s2)] = intersect


# In[14]:


#plotting intersections
sports_vals = sport_list
sports_vals_2 = sport_list
sp_sp = []

for sp in sports_vals:
    sp_int = []
    for sp2 in sports_vals_2:
        sp_int.append(sport_intersections[(sp,sp2)])
    sp_sp.append(sp_int)

sp_sp = np.array(sp_sp)
# print(sp_sp)

fig, ax = plt.subplots(figsize=(14, 14))
im = ax.imshow(sp_sp)

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(sports_vals_2)), labels=sports_vals_2)
ax.set_yticks(np.arange(len(sports_vals)), labels=sports_vals)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(sports_vals)):
    for j in range(len(sports_vals_2)):
        text = ax.text(j, i, sp_sp[i, j], ha="center", va="center", color="w")

ax.set_title("Intersection Between Sport Types (col=row set to length of players in the sport default)")
fig.tight_layout()
#plt.show()
plt.savefig('sport-sport')


# In[15]:


plt.figure(figsize=(20, 20))
bw = sns.boxplot(data=sport_times, showfliers=False)
bw.set(title='Duration Distribution Per Sport (hours)', xlabel= 'Sport', ylabel= 'Duration (hours)')
bw.tick_params(axis='x', labelrotation= 90)
plt.savefig('Duration Distribution Per Sport (hours)')


# In[16]:


plt.figure(figsize=(20, 20))
bw = sns.boxplot(data=sport_avghr, showfliers=False)
bw.set(title='Heart Rate Distribution Per Sport', xlabel= 'Sport', ylabel= 'Heart Rate')
bw.tick_params(axis='x', labelrotation= 90)
plt.savefig('HRSport')


# In[10]:


def feat(user, sport):
    #average tar heart rate, average duration, altitude gradient, gender, average user speed, average sport speed, time, sport popularity
    utar = np.median(user_avghr[user]) # user median heart rate
    star = np.median(sport_avghr[sport]) # sport median heart rate
    #print(user, sport, ': ', utar,star)

    udur = np.mean(user_times[user])  # average workout duration for the user
    sdur = np.mean(sport_times[sport]) # average workout duration for the sport
    #print(user, sport, ': ', udur, sdur)

    gen = user_gender[user] # one-hot vector gender
    if gen == 'unknown':
        gen = [0, 0]
    elif gen == 'male':
        gen = [0, 1]
    else:
        gen = [1, 0]

# altitude change median 
    u_alt_change = np.median(user_alt_change[user])
    sp_alt_change = np.median(sport_alt_change[sport])

# mode workout month
    sp_mode = st.mode(sport_start[sport])
    us_mode = st.mode(user_start[user])

    sp_mon = [0]*12
    sp_mon[sp_mode - 1] = 1


    us_mon = [0]*12
    us_mon[sp_mode - 1] = 1

# sport popularity
    sprt_pop = sport_pop[sport]

#avg speeds
    us_speed = np.median(user_speed[user])
    sp_speed = np.mean(sport_speed[sport])

    return [1] + [utar] + [udur] + [u_alt_change] + us_mon[1:] + [us_speed] + [star] + [sdur] + [sp_alt_change] + sp_mon[1:] + [sprt_pop] + [sp_speed]  + gen


# In[11]:


X_Train = [feat(d[0], d[1]) for d in x_train]
X_Test = [feat(d[0], d[1]) for d in x_test]


# In[19]:


mod = linear_model.LogisticRegression(max_iter = 1000) 
mod.fit(X_Train, y_train)

y_predict_train = mod.predict(X_Train)
mse_train = mean_squared_error(y_predict_train, y_train)
print(mse_train)
accuracy_train = sum(y_predict_train == y_train)/len(y_train)
print(accuracy_train)

y_predict = mod.predict(X_Test)
mse_test = mean_squared_error(y_predict, y_test)
print(mse_test)
accuracy_test = sum(y_predict == y_test)/len(y_test)
print(accuracy_test)


# In[20]:


# Diagnostic code blocks to check which sports are getting more misclassified

i = 0
errored_train = defaultdict()
errored_test = defaultdict()

sprt_count_train = defaultdict()
sprt_count_test = defaultdict()

for d in x_train:
    #print(d[0], d[1], y_train[i], y_predict_train[i])
    if d[1] not in sprt_count_train.keys():
        sprt_count_train[d[1]] = 0
    
    sprt_count_train[d[1]] += 1

    if y_train[i] != y_predict_train[i]:
        if d[1] not in errored_train.keys():
            errored_train[d[1]] = 0
        errored_train[d[1]] += 1
    i += 1


i = 0
for d in x_test:
    #print(d[0], d[1], y_test[i], y_predict[i])
    if d[1] not in sprt_count_test.keys():
        sprt_count_test[d[1]] = 0
    sprt_count_test[d[1]] += 1


    if y_test[i] != y_predict[i]:
        if d[1] not in errored_test.keys():
            errored_test[d[1]] = 0
        errored_test[d[1]] += 1
    i += 1


# In[21]:


for k,v in errored_test.items():
    print(k, sprt_count_test[k], v )

# %%
print(sprt_count_test)


# %%
for k, v in errored_train.items():
    print(k, v, sprt_count_train[k])
    errored_train[k] = v / sprt_count_train[k]
    print(k, errored_train[k])



# %%
for k, v in errored_test.items():
    print(k, v, sprt_count_test[k])
    errored_test[k] = v / sprt_count_test[k]
    print(k, errored_test[k])


# In[22]:


errored_train = dict(sorted(errored_train.items()))
plt.xlabel("Sport")
plt.ylabel("# of Normalized Errors")
plt.title("Sport Miscategorization (Train)")
plt.bar(errored_train.keys(), errored_train.values(), width=0.5)
plt.xticks(rotation=90)
plt.savefig('Sport Play Classification Error', bbox_inches="tight")


# In[23]:


errored_test = dict(sorted(errored_test.items()))
plt.figure()
plt.xlabel("Sport")
plt.ylabel("# of Normalized Errors")
plt.title("Sport Miscategorization (Test)")
plt.bar(errored_test.keys(), errored_test.values(), width=0.5)
plt.xticks(rotation=90)
plt.savefig('SportClassificationError(Test)', bbox_inches="tight")


# In[ ]:





# # Random Forest

# In[12]:


#########################################################Dataset Creation for Random Forest#############################################
data_train = [X_Train, y_train] 
data_test = [X_Test, y_test]


# In[76]:


rf_data = pd.DataFrame(X_Train)


# In[77]:


rf_target = pd.DataFrame(y_train)
rf_target


# In[78]:


rf_test_x = pd.DataFrame(X_Test)


# In[79]:


rf_test_y = pd.DataFrame(y_test)


# In[80]:


rf_data.columns


# In[30]:


sel = SelectFromModel(RandomForestClassifier(n_estimators = len(rf_data), random_state=42))
sel.fit(rf_data, rf_target.values.ravel())


# In[31]:


sel.get_support()


# In[32]:


selected_feat= rf_data.columns[(sel.get_support())]
len(selected_feat)


# In[33]:


rf_classifier = RandomForestClassifier(n_estimators=len(rf_data), random_state=42)
rf_classifier.fit(rf_data, rf_target.values.ravel())
feature_importances = rf_classifier.feature_importances_


# In[118]:


plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importances)), feature_importances)
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.xticks(range(len(feature_importances)), rf_data.columns, rotation=45, ha='right')  # Use feature names as x-axis labels
plt.tight_layout()
plt.show()

plt.savefig('Sport Play Classification Error', bbox_inches="tight")


# In[119]:


feature_importances


# In[42]:


sorted_indices = feature_importances.argsort()[::-1]
sorted_indices


# In[110]:


feature_importances.argsort()[::-1][:9]


# In[124]:


plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importances[:9])), feature_importances[feature_importances.argsort()[::-1][:9]])
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.xticks(range(len(feature_importances[:9])), ["Median Heart Rate Per User", "Average User Workout Duration", "Median Altitude Change Per User", "Average User Speed", "Median Heart Rate Per Sport", "Average Workout Duration Per Sport", "Median Altitude Change Per Sport", "Sport Popularity", "Average Sport Speed"], rotation=45, ha='right')  # Use feature names as x-axis labels
plt.tight_layout()

plt.savefig('Feature_Importances', bbox_inches="tight")


# In[81]:


rf_selected_train = rf_data[selected_feat]
rf_selected_test = rf_test_x[selected_feat]


# In[82]:


rf_classifier_selected = RandomForestClassifier(n_estimators=len(rf_selected_train), random_state=42)
rf_classifier_selected.fit(rf_selected_train, rf_target.values.ravel())


# In[83]:


# Make predictions on the test set
y_pred = rf_classifier_selected.predict(rf_selected_test)

# Evaluate the model performance
accuracy = sklearn.metrics.accuracy_score(rf_test_y, y_pred)
accuracy


# In[ ]:


mse = sklearn.metrics.mean_squared_error(rf_test_y, y_pred)


# In[88]:


rf_data_selected = rf_data[sorted_indices]
rf_data_selected


# In[89]:


rf_test_x = rf_test_x[sorted_indices]


# In[90]:


rf_data_selected.iloc[:, :1]


# In[92]:


rf_selected_train_1 = rf_data_selected.iloc[:, :1]

rf_selected_test_1 = rf_test_x.iloc[:, :1]

rf_classifier_selected = RandomForestClassifier(n_estimators=len(rf_selected_train_1), random_state=42)
rf_classifier_selected.fit(rf_selected_train_1, rf_target.values.ravel())

y_pred_1 = rf_classifier_selected.predict(rf_selected_test_1)

# Evaluate the model performance
accuracy_1 = sklearn.metrics.accuracy_score(rf_test_y, y_pred_1)
mse_1 = sklearn.metrics.mean_squared_error(rf_test_y, y_pred_1)
accuracy_1, mse_1


# In[97]:


rf_selected_train_2 = rf_data_selected.iloc[:, :2]

rf_selected_test_2 = rf_test_x.iloc[:, :2]

rf_classifier_selected = RandomForestClassifier(n_estimators=len(rf_selected_train_2), random_state=42)
rf_classifier_selected.fit(rf_selected_train_2, rf_target.values.ravel())

y_pred_2 = rf_classifier_selected.predict(rf_selected_test_2)

# Evaluate the model performance
accuracy_2 = sklearn.metrics.accuracy_score(rf_test_y, y_pred_2)
mse_2 = sklearn.metrics.mean_squared_error(rf_test_y, y_pred_2)
accuracy_2, mse_2


# In[98]:


rf_selected_train_3 = rf_data_selected.iloc[:, :3]

rf_selected_test_3 = rf_test_x.iloc[:, :3]

rf_classifier_selected = RandomForestClassifier(n_estimators=len(rf_selected_train_3), random_state=42)
rf_classifier_selected.fit(rf_selected_train_3, rf_target.values.ravel())

y_pred_3 = rf_classifier_selected.predict(rf_selected_test_3)

# Evaluate the model performance
accuracy_3 = sklearn.metrics.accuracy_score(rf_test_y, y_pred_3)
mse_3 = sklearn.metrics.mean_squared_error(rf_test_y, y_pred_3)
accuracy_3, mse_3


# In[100]:


rf_selected_train_4 = rf_data_selected.iloc[:, :4]

rf_selected_test_4 = rf_test_x.iloc[:, :4]

rf_classifier_selected = RandomForestClassifier(n_estimators=len(rf_selected_train_4), random_state=42)
rf_classifier_selected.fit(rf_selected_train_4, rf_target.values.ravel())

y_pred_4 = rf_classifier_selected.predict(rf_selected_test_4)

# Evaluate the model performance
accuracy_4 = sklearn.metrics.accuracy_score(rf_test_y, y_pred_4)
mse_4 = sklearn.metrics.mean_squared_error(rf_test_y, y_pred_4)
accuracy_4, mse_4


# In[102]:


rf_selected_train_5 = rf_data_selected.iloc[:, :5]

rf_selected_test_5 = rf_test_x.iloc[:, :5]

rf_classifier_selected = RandomForestClassifier(n_estimators=len(rf_selected_train_5), random_state=42)
rf_classifier_selected.fit(rf_selected_train_5, rf_target.values.ravel())

y_pred_5 = rf_classifier_selected.predict(rf_selected_test_5)

# Evaluate the model performance
accuracy_5 = sklearn.metrics.accuracy_score(rf_test_y, y_pred_5)
mse_5 = sklearn.metrics.mean_squared_error(rf_test_y, y_pred_5)
accuracy_5, mse_5


# In[104]:


rf_selected_train_6 = rf_data_selected.iloc[:, :6]

rf_selected_test_6 = rf_test_x.iloc[:, :6]

rf_classifier_selected = RandomForestClassifier(n_estimators=len(rf_selected_train_6), random_state=42)
rf_classifier_selected.fit(rf_selected_train_6, rf_target.values.ravel())

y_pred_6 = rf_classifier_selected.predict(rf_selected_test_6)

# Evaluate the model performance
accuracy_6 = sklearn.metrics.accuracy_score(rf_test_y, y_pred_6)
mse_6 = sklearn.metrics.mean_squared_error(rf_test_y, y_pred_6)
accuracy_6, mse_6


# In[106]:


rf_selected_train_7 = rf_data_selected.iloc[:, :7]

rf_selected_test_7 = rf_test_x.iloc[:, :7]

rf_classifier_selected = RandomForestClassifier(n_estimators=len(rf_selected_train_7), random_state=42)
rf_classifier_selected.fit(rf_selected_train_7, rf_target.values.ravel())

y_pred_7 = rf_classifier_selected.predict(rf_selected_test_7)

# Evaluate the model performance
accuracy_7 = sklearn.metrics.accuracy_score(rf_test_y, y_pred_7)
mse_7 = sklearn.metrics.mean_squared_error(rf_test_y, y_pred_7)
accuracy_7, mse_7


# In[125]:


rf_selected_train_8 = rf_data.iloc[:, :8]

rf_selected_test_8 = rf_test_x.iloc[:, :8]

rf_classifier_selected = RandomForestClassifier(n_estimators=len(rf_selected_train_8), random_state=42)
rf_classifier_selected.fit(rf_selected_train_8, rf_target.values.ravel())

y_pred_8 = rf_classifier_selected.predict(rf_selected_test_8)

# Evaluate the model performance
accuracy_8 = sklearn.metrics.accuracy_score(rf_test_y, y_pred_8)
mse_8 = sklearn.metrics.mean_squared_error(rf_test_y, y_pred_8)
accuracy_8, mse_8


# In[ ]:




