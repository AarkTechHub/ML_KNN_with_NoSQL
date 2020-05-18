
# coding: utf-8




import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
import matplotlib.pyplot as plt 
import seaborn as sns 
import pymongo
from pymongo import MongoClient

#Read from Mongo
conection=MongoClient("mongodb://localhost:27017/")
db=conection.Iris
data=db.irisdata
df = pd.DataFrame(list(data.find()))


# view the relationships between variables; color code by species type
sns.pairplot(df.drop(labels=['Id'], axis=1), hue='Species')

# split data into training and test sets; set random state to 0 for reproducibility 
X_train, X_test, y_train, y_test = train_test_split(df[['SepalLengthCm', 'SepalWidthCm','PetalLengthCm', 'PetalWidthCm']],df['Species'], random_state=0) 

# see how data has been split
print("X_train shape: {}\ny_train shape: {}".format(X_train.shape, y_train.shape))
print("X_test shape: {}\ny_test shape: {}".format(X_test.shape, y_test.shape))


error = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')

# initialize the Estimator object
knn = KNeighborsClassifier(n_neighbors=1)

# fit the model to training set in order to predict classes
knn.fit(X_train, y_train)

# create a prediction array for our test set
y_pred = knn.predict(X_test)

# based on the training dataset, our model predicts the following for the test set:
pd.concat([X_test, y_test, pd.Series(y_pred, name='Predicted', index=X_test.index)],ignore_index=False, axis=1)

# what is our score?
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))








