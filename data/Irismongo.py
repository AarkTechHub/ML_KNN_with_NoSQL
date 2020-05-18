import csv
import json
import pandas as pd
import sys, getopt, pprint
from pymongo import MongoClient
#CSV to JSON Conversion
csvfile = open('Iris.csv', 'r')
reader = csv.DictReader( csvfile )
mongo_client=MongoClient() 
db=mongo_client.Iris
db.segment.drop()
header= ["Id", "SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species"]

for each in reader:
    row={}
    for field in header:
        row[field]=each[field]

    db.segment.insert_one(row)