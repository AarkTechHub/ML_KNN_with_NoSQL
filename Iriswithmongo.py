

import math
import operator
import sys
import pymongo
import pandas as pd
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pymongo import MongoClient
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

conection=MongoClient("mongodb://localhost:27017/")
db=conection.Iris
data=db.irisdata
df = pd.DataFrame(list(data.find()))


def handleDataset( split,trainingSet=[] , testSet=[]):
    conection=MongoClient("mongodb://localhost:27017/")
    db=conection.pankaj
    data=db.irisdata
    df = pd.DataFrame(list(data.find()))
    Row_list =[] 
    for index, rows in df.iterrows(): 
    # Create list for the current row 
        my_list =[rows.Id, rows.SepalLengthCm, rows.SepalWidthCm, rows.PetalLengthCm, rows.PetalWidthCm, rows.Species] 
      
    # append the list to the final list 
        Row_list.append(my_list) 
    
    for x in range(len(Row_list)):
        for y in range(4):
            Row_list[x][y] = float(Row_list[x][y])
            if random.random() < split:
                trainingSet.append(Row_list[x])
            else:
                testSet.append(Row_list[x])


def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

 
def getKNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if str(testSet[x][-1]) == str(predictions[x]):
            correct += 1
    return (correct/float(len(testSet))) * 100.0

def main():
    trainingSet=[]
    testSet=[]
    split=0.67
    handleDataset(split, trainingSet, testSet)
    print ('Train: ' + repr(len(trainingSet)))
    print ('Test: ' + repr(len(testSet)))
    
    predictions=[]
    k=3
    for x in range(len(testSet)):
        neighbors=getKNeighbors(trainingSet, testSet[x], k)
        result=getResponse(neighbors)
        predictions.append(result)
        
        print('pridicted='+ repr(result) + ',actual=' + repr(testSet[x][-1]))
       
    accuracy= getAccuracy(testSet, predictions)
    print('Accuracy:' + repr(accuracy) +'%')
    
        





main()

features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']# Separating out the features
x = df.loc[:, features].values



y = df.loc[:,['Species']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents,columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, df[['Species']]], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('Iris set', fontsize = 20)
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Species'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()




plt.figure(figsize = (10, 7)) 
x = df["SepalLengthCm"] 

plt.hist(x, bins = 20, color = "green") 
plt.title("Sepal Length in cm") 
plt.xlabel("Sepal_Length_cm") 
plt.ylabel("Count") 





new_data = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]] 
print(new_data.head())

plt.figure(figsize = (10, 7)) 
new_data.boxplot() 







