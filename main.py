#!/usr/bin/python
# Author: Jasjeet Dhaliwal
# Date: 6/10/2017

import os, sys, gc
from database import Database
from lda import LDA
from argparse import ArgumentParser

def run_lda(data_dir, num_topics, use_mini_batches, batch_size, epochs,visualize, model_file, create_dict, dict_file):
  """ Run training and display test results if visualize is true

  Args:
    data_dir(str): directory containing director(y/ies) of data
    num_topics(int): Number of topics to train the model on
    batch_size(int): Size of mini batches used to train the model
    epochs(int): Number of epochs to train the data for on the train set
    visualize(bool): visuallize results of the trained model
  """
   
  assert(os.path.isdir(data_dir)), "Invalid data directory path"

  use_model_file = False
  if model_file:
    use_model_file = True


  #Create model
  lda = LDA(num_topics=num_topics)
  if create_dict:
    #Create word to id mapping for all texts 
    lda.create_dict(data_dir)
    lda.store_dict_to_disk('./dict_'+ data_dir)
  else:
    lda.load_dict_from_disk(dict_file)
  
  for root, dirs, files in os.walk(data_dir):
    #Iterate over sub-dirs
    for d in dirs:
        #Create database object
        db = Database(d, os.path.abspath(data_dir+'/'+d))
        #Add database to model 
        lda.add_database(db)  

        if use_model_file:
          #Load model paramaters from model file and call train
          lda.train(model_file,db_name=db.get_name(), use_mini_batches=use_mini_batches, use_internal_dict=True,batch_size=batch_size, num_epochs=epochs)
          #Set to false, as we just need to load the model once and train it on the entire dataset
          use_model_file = False
        else:
          #Call train on the model
          lda.train(db_name=db.get_name(), use_mini_batches=use_mini_batches, use_internal_dict=True, batch_size=batch_size, num_epochs=epochs)

        #Remove db to free memory (can also save it if preferred)
        db.store_to_disk('./databases/' + d)
        lda.remove_database(db.get_name())
        del db
        gc.collect()
        tmp_file = './models/' + d + str(num_topics)
        lda.save_model(tmp_file)
  
  #Save final model
  file_name = './models/final' + str(num_topics)
  lda.save_model(file_name)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument("data_directory", help="Provide the path to the data directory", type=str)
  parser.add_argument("--use_mini_batches", help="bool for training on mini batches or entire dataset", action='store_true')
  parser.add_argument("--batch_size", help="Provide the batch_size during training", default=1000, type=int)
  parser.add_argument("--num_epochs", help="Provide the number of epochs to train for", default=1, type=int)
  parser.add_argument("--num_topics", help="Provide the number of topics to run lda", default=6, type=int)
  parser.add_argument("--model_file", help="Provide the path to the model object that needs to be trained", default='', type=str)
  parser.add_argument("--dict_file", help="Provide the path to the dict object that needs to be loaded", default='', type=str)
  parser.add_argument("--visualize", help="1 means visualize training and results, 0 means don't visualize", action='store_true')
  parser.add_argument("--create_dict", help="Load a dictionary or create a new dictionary", action='store_false')
  args = parser.parse_args()
  run_lda(args.data_directory, args.num_topics, args.use_mini_batches, args.batch_size, args.num_epochs, args.visualize, args.model_file, args.create_dict, args.dict_file)
