import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

train = pd.read_csv('/Users/zhangyuan/Documents/MLCtmp/train.csv')
test = pd.read_csv('/Users/zhangyuan/Documents/MLCtmp/test.csv')
#--------------------------------------
train_eeg=train.loc[:,'eeg_0':'eeg_1999']
train_resp=train.loc[:,'respiration_x_0':'respiration_z_399']
train_other=train.loc[:,'time_previous':'night']
# EGG----------------------------------
train_eeg_mean=np.mean(np.array(train_eeg))
train_eeg_std=np.std(np.array(train_eeg))
#what we want
train_eeg_normal=(train_eeg-train_eeg_mean)/train_eeg_std
test_eeg=test.loc[:,'eeg_0':'eeg_1999']
test_eeg_normal=(test_eeg-train_eeg_mean)/train_eeg_std
#clean
del train_eeg,test_eeg


# Respiration---------------------------
train_resp_x=train_resp.iloc[:,list(range(0, train_resp.shape[1]-1, 3))]
train_resp_y=train_resp.iloc[:,list(range(1, train_resp.shape[1]-1, 3))]
train_resp_z=train_resp.iloc[:,list(range(2, train_resp.shape[1]-1, 3))]
#
train_resp_x_mean=np.mean(np.array(train_resp_x))
train_resp_x_std=np.std(np.array(train_resp_x))
train_resp_y_mean=np.mean(np.array(train_resp_y))
train_resp_y_std=np.std(np.array(train_resp_y))
train_resp_z_mean=np.mean(np.array(train_resp_z))
train_resp_z_std=np.std(np.array(train_resp_z))
#what we want
train_resp_x_normal=(train_resp_x-train_resp_x_mean)/train_resp_x_std
train_resp_y_normal=(train_resp_y-train_resp_y_mean)/train_resp_y_std
train_resp_z_normal=(train_resp_z-train_resp_z_mean)/train_resp_z_std
#
test_resp=test.loc[:,'respiration_x_0':'respiration_z_399']
test_resp_x=test_resp.iloc[:,list(range(0, test_resp.shape[1]-1, 3))]
test_resp_y=test_resp.iloc[:,list(range(1, test_resp.shape[1]-1, 3))]
test_resp_z=test_resp.iloc[:,list(range(2, test_resp.shape[1]-1, 3))]
#what we want
test_resp_x_normal=(test_resp_x-train_resp_x_mean)/train_resp_x_std
test_resp_y_normal=(test_resp_y-train_resp_y_mean)/train_resp_x_std
test_resp_z_normal=(test_resp_z-train_resp_z_mean)/train_resp_x_std
#clean
del train_resp,train_resp_x,train_resp_y,train_resp_z,test_resp,test_resp_x,test_resp_y,test_resp_z
# Other data---------------------------
scaler = StandardScaler()
scaler.fit(train_other)
train_other_normalize=pd.DataFrame(scaler.transform(train_other),columns=['time_previous','number_previous','time','user','night'])

test_other=test.loc[:,'time_previous':'night']
test_other_normalize = pd.DataFrame(scaler.transform(test_other),columns=['time_previous','number_previous','time','user','night'])
#clean
del train_other,test_other
# Output
train_output=train.loc[:,'power_increase']
del train,test
# All data
train_data = pd.concat((train_eeg_normal,
                      train_resp_x_normal,
                      train_resp_y_normal,
                      train_resp_z_normal,
                      train_other_normalize,
                      train_output),axis=1)

test_data = pd.concat((test_eeg_normal,
                      test_resp_x_normal,
                      test_resp_y_normal,
                      test_resp_z_normal,
                      test_other_normalize),axis=1)

pickle.dump(train_data, open('/Users/zhangyuan/Documents/MLCtmp/pyMLC/train_data.sav', 'wb'))
pickle.dump(test_data, open('/Users/zhangyuan/Documents/MLCtmp/pyMLC/test_data.sav', 'wb'))
#countvectorizer_uni=pickle.load(open('countvectorizer_uni_k_unigram.sav', 'rb'))
del train_data,test_data
train_data_woresp = pd.concat((train_eeg_normal,
                      train_other_normalize,
                      train_output),axis=1)

test_data_woresp = pd.concat((test_eeg_normal,
                      test_other_normalize),axis=1)
pickle.dump(train_data_woresp, open('/Users/zhangyuan/Documents/MLCtmp/pyMLC/train_data_woresp.sav', 'wb'))
pickle.dump(test_data_woresp, open('/Users/zhangyuan/Documents/MLCtmp/pyMLC/test_data_woresp.sav', 'wb'))
