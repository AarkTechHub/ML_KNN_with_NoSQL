import os
import sklearn.metrics as metrics
import sklearn
import pymongo
import pickle
import time
from pymongo import MongoClient
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


#FUNCTIONs ... BEGIN
"""
"""
def save_model_to_db(model, client, db, dbconnection, model_name):
    pickled_model = pickle.dumps(model)
    # creating connection
    myclient = pymongo.MongoClient(client)
    # creating database in mongodb mydb myclient[db)
    mydb = myclient[db]
    # creating collection
    mycon = mydb[dbconnection]
    info = mycon.insert_one({model_name: pickled_model, 'name': model_name, 'created_time': time.time()})
    print(info.inserted_id, "saved with this id successfullyt")
    details = {
        'inserted_id': info.inserted_id,
        "model_name": model_name,
        'created_time': time.time()
    }
    return details




"""
"""
def load_saved_model_from_db(model_name, client, db, dbconnection):
    json_data = {}
    myclient = pymongo.MongoClient(client)
    # creating database in mongodb mydb myclient[db)
    mydb = myclient[db]
    # creating collection
    mycon = mydb[dbconnection]
    data = mycon.find({'name': model_name})

    for i in data:
        json_data = i
    # fetching model from_db
    pickled_model = json_data[model_name]
    return pickle.loads(pickled_model)

#FUNCTIONS ... END



#MAIN CODE

conection = MongoClient("mongodb://localhost:27017/")
db = conection.car
data = db.segment
df = pd.DataFrame(data.find({}, {'_id': False}))

df['class'], class_names = pd.factorize(df['class'])

df['buying'], _ = pd.factorize(df['buying'])
df['maint'], _ = pd.factorize(df['maint'])
df['doors'], _ = pd.factorize(df['doors'])
df['persons'], _ = pd.factorize(df['persons'])
df['lug_boot'], _ = pd.factorize(df['lug_boot'])
df['safety'], _ = pd.factorize(df['safety'])

X = df.iloc[:, :-1]
y = df.iloc[:, -1]  # split data randomly into 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["car"]
mycol = mydb["acc"]

for K in range(25):
    K_value = K + 1
    neigh = KNeighborsClassifier(n_neighbors=K_value)
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    mydict = {"Accuracy": acc, "kvalue": K_value}
    x = mycol.insert_one(mydict)

kval = mydb.acc.find_one(sort=[("Accuracy", pymongo.DESCENDING)])
maxacc = kval["kvalue"]
print("value of k is", maxacc)

# train the decision tree
## Instantiate the model with 5 neighbors.
model = KNeighborsClassifier(n_neighbors=maxacc)
## Fit the model on the training data.
model.fit(X_train, y_train)


details = save_model_to_db(model=model, client='mongodb://localhost:27017/', db='car', dbconnection='data',
                           model_name='carmodel')

x = load_saved_model_from_db(model_name=details['model_name'], client='mongodb://localhost:27017/', db='car',
                             dbconnection='data')

print(x.score(X_test, y_test))


