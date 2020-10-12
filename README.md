# Anomaly-Detection-and-Prediction-of-Respiratory-Rate

Effective monitoring and alarm systems are increasingly needed in health care service because of growing aged population. These systems (e.g., vital sign monitoring sensors) help to monitor the health status of users (patients). 

The abnormal signal from these biomonitoring devices may indicate critical change in the health status of patients. Therefore, anomaly detection of sensor data is important for providing instant health care services to patients. 

In this study, I examined the sensor data on the respiration rate of patients provided by Keenly Health and explored the methods to detect and predict the anomaly among the respiration rate data. 

First, I cleaned data and aggregated the data into hourly data, then applied two methods for anomaly detection: one is seasonal-trend decomposition and the other is K-means clustering.  

Based on the results for anomaly detection, I labelled each data point into normal vs abnormal, then applied nine machine learning algorithms, including Naive Bayes, Linear discriminant analysis (LDA), logistic regression, K Nearest Neighbor (KNN), Support Vector Machine (SVM), Neural Network, Decision Tree, Random Forest,  and Adaboost to predict the anomalies. 

The results showed these algorithms performed very well, with an overall accuracy of 99% approximately. However, the Area Under Curve (AUC) of receiver operating characteristic (ROC) varied among different algorithms and the methods for anomaly detection. Overall, the Adaboost algorithm performed best, with the average AUC of 96%.



Key words: Anomaly detection, machine learning, seasonal-trend decomposition, respiration rate, health care

