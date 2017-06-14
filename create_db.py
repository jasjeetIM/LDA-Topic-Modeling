#!/usr/bin/python
# Author: Jasjeet Dhaliwal

import os, sys, gc, pickle
from database import Database

#Create a database object from a subdirectory
#and store to disk for faster training in the future
#Ex:
db = Database('PMC_11', os.path.abspath('./data/PMC_11'))
db.store_to_disk('./databases/PMC_11')
del db

