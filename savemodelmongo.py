import time
import pickle
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
import matplotlib.pyplot as plt 
import seaborn as sns 
import pymongo
from pymongo import MongoClient

conection=MongoClient("mongodb://localhost:27017/")
db=conection.Iris3
data=db.segment
df = pd.DataFrame(list(data.find()))

# split data into training and test sets; set random state to 0 for reproducibility 
X_train, X_test, y_train, y_test = train_test_split(df[['SepalLengthCm', 'SepalWidthCm','PetalLengthCm', 'PetalWidthCm']],df['Species'], random_state=0) 


# see how data has been split
print("X_train shape: {}\ny_train shape: {}".format(X_train.shape, y_train.shape))
print("X_test shape: {}\ny_test shape: {}".format(X_test.shape, y_test.shape))

# initialize the Estimator object
knn = KNeighborsClassifier(n_neighbors=1)


# fit the model to training set in order to predict classes
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# based on the training dataset, our model predicts the following for the test set:
pd.concat([X_test, y_test, pd.Series(y_pred, name='Predicted', index=X_test.index)],ignore_index=False, axis=1)

# what is our score?
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))

model = knn# put yours model
#model.fit(X_train, y_train)

# save the model to disk
filename = 'final.sav'

pickle.dump(model, open(filename, 'wb'))




# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)


def save_model_to_db(model,client,db,dbconnection,model_name):
    pickled_model=pickle.dumps(model)
# creating connection
    myclient=pymongo.MongoClient(client)
#creating database in mongodb mydb myclient[db)
    mydb=myclient[db]
#creating collection
    mycon=mydb[dbconnection]
    info=mycon.insert_one({model_name: pickled_model, 'name': model_name, 'created_time':time.time()})
    print(info.inserted_id,"saved with this id successfullyt")
    details={
    'inserted_id':info.inserted_id,
    "model_name":model_name,
    'created_time':time.time()
    }
    return details

details=save_model_to_db(model=knn,client='mongodb://localhost:27017/',db='Iris3',dbconnection='data1',model_name='final1')

def load_saved_model_from_db(model_name,client,db,dbconnection):
    json_data={}
    myclient=pymongo.MongoClient(client)
#creating database in mongodb mydb myclient[db)
    mydb=myclient[db]
#creating collection
    mycon=mydb[dbconnection]
    data=mycon.find({'name':model_name})
    
    for i in data:
        json_data=i
    #fetching model from_db
    pickled_model=json_data[model_name]
    return pickle.loads(pickled_model)
    
        
    
    

    
    


# In[ ]:


x=load_saved_model_from_db(model_name=details['model_name'],client='mongodb://localhost:27017/',db='Iris3',dbconnection='data1')


# In[ ]:


print(x.score(X_test, y_test)


# In[ ]:





# In[27]:





# In[28]:





# In[29]:





# In[30]:





# In[ ]:





# In[ ]:





# In[ ]:




