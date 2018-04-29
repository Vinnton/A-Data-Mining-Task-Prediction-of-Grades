# Data Mining Task: Prediction of Grades
Clickstream is important for mining user's latent behavior. As for online learning platform, the lectuerer can monitor students' learning pattern by clickstream pattern analysis. Video-clickstream records students' click actions when watching lecture videos. A general video-clickstream log file contains the following events: load_video, play_video, pause_video, seek_video, stop_video, and speed_change_video.

In this part, the project predicts students' final exam performance, passed or failed, based on their clickstream event log from 63 lecture videos in the same online learning semester.

The dataset including:
1. TrainFeatures.csv: 5050 students' video clickstream log info covering 63 videos. You need to use it for training.
2. TestFeatures.csv: 1293 students' video clickstream log info covering 63 videos. You need to use it in the testing stage. The students in TrainFeatures.csv and TestFeatures.csv belong to the same learning semester and share the same grading strategy. Since two files above are too large to upload to Github, please download them from https://drive.google.com/open?id=1rN1d-5Z53xjxFDuMOJ5N8Le_2O0emWEB.
3. TrainLabel.csv: The label for 5050 students (1 for pass, 0 for fail).
4. TestData.csv: The students you need to give prediction for. Their learning log info can be found in TestFeatures.csv.
5. VideoInfo.csv: Video duration info for 63 lecture videos.
6. Sample_submission.csv: The sample submission file you may refer.
7. Description.pdf: Some description for the log events and attributes.

The idea to realize the project and how to run thw code:
Coding environment: Python 3.6, IDE: Pycharm
Steps of the solution:
1.  Preprocessing of the data and features generation
a. Mapping the long string user_id/ video_id into simple number id;
b. Delete the attribute session since the attribute do nothing to the results;
c. Calculating the operation times on each video for each user, for example, if user A loaded and then played video 1, then the operation times is 2. Hence the attribute new_speed, old_speed, new_time, old_time are already considered in this situation;
d. Generate a new file TrainDataFinal.csv to store the data.
After the steps above, we could have a table as below; Here V1 stands for Video1. Same preprocessing period is also implemented on TestFeatures.csv, and a new TestDataFinal.csv would be generated. 
2. Classification method: Several classification methods are compared as below, with a cross_validation of 4:1.
3. How to run the code:
a. Firstly, put all the data files into the directory where contains the python files;
b. Run TrainDataPreprocessing.py and TestDataPreprocessing.py first to generate TrainDataFinal.csv and TestDataFinal.csv. The reason of separating preprocessing and training process is to prevent time wasting while changing models and factors in training process;
c. Finally run Train.py to generate accuracy of each classification method and a prediction of test data prediction.csv.
