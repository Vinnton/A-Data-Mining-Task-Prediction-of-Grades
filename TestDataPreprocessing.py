import numpy as np
import pandas as pd
from sklearn import preprocessing
import scipy
import csv

data3 = []
with open('TestFeatures.csv') as f3:
    fid3 = csv.reader(f3)
    for row in fid3:
        data3.append(row[0:])
data3.pop(0)

TestData = np.array(data3)
le = preprocessing.LabelEncoder()

'''
# Step 1:  Mapping the long string user id/ video id into simple number id
# Step 2:  Delete the attribute session since the attribute do nothing to the results 
'''
TestData = scipy.delete(TestData,[2,3,4,5,6],1) # delete insignificant attributes
le.fit(TestData[:,0]) # Map the user_id into numeric id
TestData[:,0] = le.transform(TestData[:,0])
le.fit(TestData[:,1]) # Map the video_id into numeric id
TestData[:,1] = le.transform(TestData[:,1])


'''
# Step 3:  Calculating the operation times on each video for each user
'''
TestData = [x for x in TestData if x[2] != 'load_video'] # delete 'load_video'
TestDataFrame = pd.DataFrame(TestData,columns=['user_id','video_id','event_type','time'])
TestDataFrame['user_id'] = TestDataFrame['user_id'].astype('int') # transform type 'string' to 'int'
TestDataFrame['video_id'] = TestDataFrame['video_id'].astype('int')
TestDataFrame = TestDataFrame.sort_values(by=['user_id','video_id']) # sort the data with user_id and video_id
TestDataModified = TestDataFrame.groupby(['user_id','video_id']).size().reset_index(name='operation_times') # calculate the operation times


'''
# Step 4:  Generate the final train data
'''
TestDataFinal = TestDataModified.pivot_table(values='operation_times', index='user_id', columns='video_id', fill_value = 0) # generate the final train data
TestDataFinal = TestDataFinal.reindex(np.arange(TestDataFinal.index.min(), TestDataFinal.index.max() + 1), fill_value=0) # Fill missing columns with zeros
TestDataFinal.to_csv('TestDataFinal.csv',index=False)

