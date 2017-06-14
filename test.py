#!/usr/bin/python
# Author: Jasjeet Dhaliwal

import os, sys, gc, pickle
from database import Database
from lda import LDA
from argparse import ArgumentParser

def test_lda(model_file, dict_file, dbs_dir):
  """ Run training and display test results if visualize is true

  Args:
    model_file(str): saved model file to continue training on
    dict_file(str): dict_file path to load dictionary from 
    dbs_dir(str): dir path to load databases from 
  """
   
  assert(os.path.isdir(dbs_dir)), "Invalid data directory path"
  lda = LDA()
  print 'Loading existing dictionary...'
  lda.load_dict_from_disk(dict_file)
  test_results = list()
  #Iterate over all data and train model
  for root, dirs, files in os.walk(dbs_dir):
    #Iterate over sub-dirs
    for d in files:
        db = Database()
        #Load database object from saved file
        db.load_from_disk(data_dir + '/' + d)

        #Add database to model 
        lda.add_database(db)  
        #Test model
        test_results.append(lda.test(model_file, db_name=db.get_name()))
        lda.remove_database(db.get_name())

        del db
        gc.collect()
  
  #Print test results
  print test_results
  #Show top words for each topic 
  print lda.model.show_topics()

if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument("--model_file", help="Provide the path to the model object that needs to be trained", default='./models/PMC_46', type=str)
  parser.add_argument("--dict_file", help="Provide the path to the dict object that needs to be loaded", default='./dict/dictionary', type=str)
  parser.add_argument("--db_dir", help="Provide path to database on which to test", default='./databases/', type=str)
  args = parser.parse_args()
  test_lda(args.model_file, args.dict_file, args.load_dbs)
