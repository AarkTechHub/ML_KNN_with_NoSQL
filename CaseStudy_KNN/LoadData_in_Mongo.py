import csv
import json
import pandas as pd
import sys, getopt, pprint
import os
from pymongo import MongoClient
#CSV to JSON Conversion
path = os.getcwd() + "/data/car.data"
csvfile = open(path, 'r')
reader = csv.DictReader( csvfile )
next(reader)  # Skip header row.
mongo_client=MongoClient()
db=mongo_client.car3
db.segment.drop()
header= ['buying','maint','doors','persons','lug_boot','safety','class']

for i in range(1,7):

    for each in reader:
        row={}
        for field in header:
            row[field]=each[field]

        db.segment.insert_one(row)
    csvfile.seek(0)
    next(reader)  # Skip header row.