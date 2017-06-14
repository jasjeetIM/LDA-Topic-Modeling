#!/usr/bin/python
#Author: Jasjeet Dhaliwal

import sys, os, math, gc, codecs,pickle
from database import Database
from gensim import corpora, models
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from stop_words import get_stop_words


class LDA(object):
  """Topic model using Latent Dirichlet Allocation and collapsed Gibbs sampling"""
    
  def __init__(self, database=None, num_topics=3):
    """Train and infer on the Database class

    Args: 
      database(Database): Database object to train and test on
    """
    #The lda model
    self.model = None
    #Databases that can be used to train the model
    self.databases = dict()
    #Num of topics (hyperparam)
    self.num_topics = num_topics
    self.word2idx = None

    if database is not None:
      self.databases[database.get_name()] = database
      

  def add_database(self, database=None):
    """Add a database to the dict of databases
    
    Args:
      database(Database): Database object to train and test on
    """

    if database is not None:
      if database.get_name() in self.databases:
        print "Replacing existing database with the same name"
        self.databases[database.get_name()] = database
      else:
        self.databases[database.get_name()] = database
    else:
      print "Warning: No Database provided, nothing added."


  def remove_database(self,db_name=''):
    """Remove a database by name from the list of databases
 
    Args:
     db_name(str): name of db object to remove

    """

    if not db_name or not isinstance(db_name,str):
      print "Please provide a non empty str for the db_name arg"
      return

    else:
      if db_name in self.databases:
        del self.databases[db_name] 
        return
      
    print "Warning: No valid database name provided. Nothing deleted."

  
  def train(self, model_file='', db_name='', use_mini_batches=True, use_internal_dict=True, batch_size=1, num_epochs=1):
    """Train LDA model on a database

    Args:
     model_file(str): name of file containing an older model to load and train
     db_name(str): name of database to train on
     use_mini_batches(bool): if True, train model on mini-batches of batch size
                             if False, train model on entire training set at once
                             Note: Use False only if there is sufficient RAM.
     use_internal_dict(bool): If True, use internal dictionary 
                              If False, use dictionary generated from the database object
     batch_size(int): size of a mini-batch
     num_epochs(int): number of epochs to train model on the training set of this db

    """ 

    if not db_name or not isinstance(db_name,str):
      print "Please provide a non empty str for the db_name arg"
      return

    else:
      if db_name in self.databases:
        self.train_on_db(model_file, db_name, use_mini_batches, use_internal_dict, batch_size, num_epochs=1)
        return 
      
    print "Warning: no valid datbase name provided. No training done."


  def train_on_db(self, model_file='', db_name='',  use_mini_batches=True, use_internal_dict=True, batch_size=1, num_epochs=1):
    """Train on the given database
    Args:
     model_file(str): name of file containing an older model to load and train
     db_name(str): name of database to train on
     use_mini_batches(bool): if True, train model on mini-batches of batch size
                             if False, train model on entire training set at once
                             Note: Use False only if there is sufficient RAM.
     use_internal_dict(bool): If True, use internal dictionary 
                              If False, use dictionary generated from the database object
     batch_size(int): size of a mini-batch
     num_epochs(int): number of epochs to train model on the training set of this db
    """

    
    #Train using mini batches (Uses EM for updates)
    if use_mini_batches:

      #Train on an existing model
      if model_file:
        assert(os.path.isfile(model_file)), "Invalid model file path"
        db = self.databases[db_name]
        db.prep_train_epoch(batch_size, num_epochs)
        mini_batch = db.get_mini_batch()

        if not use_internal_dict:
          word2idx = db.get_word2id()
        else:
          word2idx = self.word2idx

        mb_number = 1
        tmp_model = models.ldamodel.LdaModel(num_topics=self.num_topics, id2word = word2idx, passes=1)
        self.model = tmp_model.load(model_file)
        while mini_batch:
          self.model.update(mini_batch)
          print ('Completed mini-batch: {}'.format(mb_number))
          mb_number+=1
          mini_batch = db.get_mini_batch()
        print ('Training has been successfully completed on database {}'.format(db_name))

     #Train on a new model
      else:

        #Train on self.model if it exists
        if self.model is not None:
          db = self.databases[db_name]
          db.prep_train_epoch(batch_size, num_epochs)
          mini_batch = db.get_mini_batch()

          mb_number = 1
          while mini_batch:
            self.model.update(mini_batch)
            print ('Completed mini-batch: {}'.format(mb_number))
            mb_number+=1
            mini_batch = db.get_mini_batch()
          print ('Training has been successfully completed on database {}'.format(db_name))

        #Generate a new self.model and start training
        else:
          db = self.databases[db_name]
          db.prep_train_epoch(batch_size, num_epochs)
          mini_batch = db.get_mini_batch()

          if not use_internal_dict:
            word2idx = db.get_word2id()
          else:
            word2idx = self.word2idx

          mb_number = 1
          self.model = models.ldamodel.LdaModel(mini_batch, num_topics=self.num_topics, id2word = word2idx, passes=1)
          print ('Completed mini-batch: {}'.format(mb_number))
          mini_batch = db.get_mini_batch()
          while mini_batch:
            self.model.update(mini_batch)
            mb_number+=1
            print ('Completed mini-batch: {}'.format(mb_number))
            mini_batch = db.get_mini_batch()
          print ('Training has been successfully completed on database {}'.format(db_name))


    #Train on entire dataset
    else:
      if model_file:
        assert(os.path.isfile(model_file)), "Invalid model file path"
        db = self.databases[db_name]

        if not use_internal_dict:
          word2idx = db.get_word2id()
        else:
          word2idx = self.word2idx

        tmp_model = models.ldamodel.LdaModel(num_topics=self.num_topics, id2word = word2idx, passes=1)
        self.model = tmp_model.load(model_file)
        self.model.update(db.get_train_set())
        print ('Training has been successfully completed on database {}'.format(db_name))

     #Train on a new model
      else:
        if self.model is not None:
          db = self.databases[db_name]
          self.model.update(db.get_train_set())
          print ('Training has been successfully completed on database {}'.format(db_name))

        else:
          db = self.databases[db_name]

          if not use_internal_dict:
            word2idx = db.get_word2id()
          else:
            word2idx = self.word2idx

          self.model = models.ldamodel.LdaModel(db.get_train_set(), num_topics=self.num_topics, id2word = word2idx, passes=1)
          print ('Training has been successfully completed on database {}'.format(db_name))

    print ('Size of model = {}'.format(sys.getsizeof(self.model)))
        

  
  def create_dict(self, data_dir):
    """Creates a dictionary object from the training data

    Args:
      data_dir(str): path to data directory
 
    Output:
      gensim.corpora.Dictionary object to be used by the model for training
    """

    stop_words = get_stop_words('en')
    stemmer = PorterStemmer()
    files_read = 0
    tokenized_texts = list()

    if data_dir is not None:
      assert(os.path.isdir(data_dir)), "Invalid data directory path"
      print ('Creating a dictionary from the directory : {}'.format(data_dir))
      for root, dirs, files in os.walk(data_dir):
        for d  in dirs:
          for sub_root, sub_dirs, sub_files in os.walk(data_dir + '/' + d):
            for f in sub_files:
              #Read in data for all .txt files
              if f.endswith('.txt'):
                with codecs.open(data_dir + '/' + d + '/' + f, 'r', 'utf-8-sig') as data_f:
                  doc = data_f.read().replace('\n', ' ')
                  #Tokenize 
                  tokens = word_tokenize(doc.lower())
                  #Remove stop words
                  stop_tokens = [token for token in tokens if token not in stop_words]
                  #Step text using Porter Stemming Algorithm
                  stem_tokens = list(set([stemmer.stem(token) for token in stop_tokens]))
                  tokenized_texts.append(stem_tokens)
                  files_read+=1

                  if not (files_read % 1000):
                    print ('Files completed : {}, Number of tokens in last file: {}'.format(files_read, len(tokenized_texts[-1])))

                  #Clear up unused variables for efficient mem usage
                  del doc
                  del tokens
                  del stop_tokens
                  del stem_tokens
                  gc.collect()
               

    if files_read > 0:
      #Assign an integer to each unique word in the texts
      self.word2idx = corpora.Dictionary(tokenized_texts)
    print "Successfully created an internal dictionary."

  def store_dict_to_disk(self, file_path):
    """Store the database object to disk for future use
    
     Args: 
      file_path(str): absolute or relative path of file to store the db in
    """
    
    assert(os.path.dirname(file_path)), 'Invalid directory provided to save file'
    assert(os.access(os.path.dirname(file_path), os.W_OK)), 'Need write permissions to parent dir'

    with open(file_path, 'w') as f:
      if self.word2idx is not None:
        pickle.dump([self.word2idx],f)


  def load_dict_from_disk(self, file_path):
    """Load the corpus from disk 
     Args: 
      file_path(str): absolute or relative path of file to store the db in
    """
    
    assert(os.path.isfile(file_path)), 'Invalid file path to load db'

    with open(file_path) as f:
      self.word2idx = pickle.load(f)


  def save_model(self, file_name=''):
    """Save model
    Args: 
      file_name(str): file to save the lda model to 

    """ 

    if self.model is not None:
      self.model.save(file_name)
    else:
      print "No model trained yet."

