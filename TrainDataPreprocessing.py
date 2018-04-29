import numpy as np
import pandas as pd
from sklearn import preprocessing
import scipy
import csv

data1 = []
data2 = []
with open('TrainFeatures.csv') as f1:
    fid1 = csv.reader(f1)
    for row in fid1:
        data1.append(row[0:])

with open('TrainLabel.csv') as f2:
    fid2 = csv.reader(f2)
    for row in fid2:
        data2.append(row[0:])
data1.pop(0)
data2.pop(0)

TrainData = np.array(data1)
TrainLabel = np.array(data2)
le = preprocessing.LabelEncoder()


'''
# Step 1:  Mapping the long string user id/ video id into simple number id
# Step 2:  Delete the attribute session since the attribute do nothing to the results 
'''
TrainData = scipy.delete(TrainData,[2,3,4,5,6],1) # delete insignificant attributes
le.fit(TrainData[:,0]) # Map the user_id into numeric id
TrainData[:,0] = le.transform(TrainData[:,0])
le.fit(TrainData[:,1]) # Map the video_id into numeric id
TrainData[:,1] = le.transform(TrainData[:,1])
le.fit(TrainLabel[:,0]) # same pre-processing on train labels
TrainLabel[:,0] = le.transform(TrainLabel[:,0])


'''
# Step 3:  Calculating the operation times on each video for each user
'''
TrainData = [x for x in TrainData if x[2] != 'load_video'] # delete 'load_video'
TrainDataFrame = pd.DataFrame(TrainData,columns=['user_id','video_id','event_type','time'])
TrainDataFrame['user_id'] = TrainDataFrame['user_id'].astype('int') # transform type 'string' to 'int'
TrainDataFrame['video_id'] = TrainDataFrame['video_id'].astype('int')
TrainDataFrame = TrainDataFrame.sort_values(by=['user_id','video_id']) # sort the data with user_id and video_id
TrainLabelFrame = pd.DataFrame(TrainLabel,columns=['user_id','grade']) # same pre-processing on train labels
TrainLabelFrame['user_id'] = TrainLabelFrame['user_id'].astype('int')
TrainLabelFrame = TrainLabelFrame.sort_values(by=['user_id'])
TrainDataModified = TrainDataFrame.groupby(['user_id','video_id']).size().reset_index(name='operation_times') # calculate the operation times


'''
# Step 4:  Generate the final train data
'''
TrainDataFinal = TrainDataModified.pivot_table(values='operation_times', index='user_id', columns='video_id', fill_value = 0) # generate the final train data
TrainDataFinal = TrainDataFinal.reindex(np.arange(TrainDataFinal.index.min(), TrainDataFinal.index.max() + 1), fill_value=0) # Fill missing columns with zeros
TrainLabelFinal = TrainLabelFrame.reset_index(drop=True)
TrainLabelFinal.pop('user_id')
# print to csv
TrainDataFinal.to_csv('TrainDataFinal.csv',index=False)
print(TrainLabelFinal)
TrainLabelFinal.to_csv('TrainLabelFinal.csv',index=False)
