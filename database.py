#! /usr/bin/python
# Author: Jasjeet Dhaliwal
# Date: 6/10/2017

import sys,os, pickle, random, math, gc, codecs
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from stop_words import get_stop_words
from gensim import corpora


class Database(object):
  """Database object to read and store text corpus (in English only)"""

  def __init__(self, db_name='', data_dir=None, test_pct=10):
    """Initialize database with all .txt files in the data directory
       (Functionality is currently limited to .txt files)

      Args: 
        data_dir (str): absolute or relative path to directory containing 
                           text files. (i.e. each text file will become 
                           a document in the database)
       test_pct(int): Percent of total data to be used as the test set
                      Must be between  0 and 100
        db_name(str): name of database

    """

    if not db_name or  not (isinstance(db_name,str)):
      self.db_name = (time.strftime("%D:%H:%M:%S"))
    else:
      self.db_name = db_name

    #Get stop words for the English language
    self.stop_words = get_stop_words('en')
    #Get stemmer 
    self.stemmer = PorterStemmer()
    #Store file stats
    self.files_read = 0
    #List of pre-processed text files
    self.tokenized_texts = list()
    #Map of words and unique ids assigned to words (required by gensim)
    self.word_to_id = None
    #Training data
    self.train_set = None
    #Test data
    self.test_set = None
    #Utility list used to segment corpus into training and test set
    self.train_epoch_idx = None
    #Batch size of mini_batches to be used from the training set
    self.batch_size = 0

    if test_pct >= 0 and test_pct <= 100:
      self.test_pct = test_pct*0.01
    else:
      self.test_pct = 0.1

    #Mem usage profiling variables
    start_idx = 0
    size = 0

    if data_dir is not None:
      assert(os.path.isdir(data_dir)), "Invalid data directory path"
      path, dirs, files = os.walk(data_dir).next()
      print ('Files to add to database: {}'.format(len(files)))
      for root, dirs, files in os.walk(data_dir):
        #Iterate over files
        for f in files:
          #Read in data for all .txt files
          if f.endswith('.txt'):
            with codecs.open(data_dir + '/' + f, 'r', 'utf-8-sig') as data_f:
              doc = data_f.read().replace('\n', ' ')
              #Tokenize 
              tokens = word_tokenize(doc.lower())
              #Remove stop words
              stop_tokens = [token for token in tokens if token not in self.stop_words]
              #Step text using Porter Stemming Algorithm
              stem_tokens = [self.stemmer.stem(token) for token in stop_tokens]
              self.tokenized_texts.append(stem_tokens)
              self.files_read+=1

              #Clear up unused variables for efficient mem usage
              del doc
              del tokens
              del stop_tokens
              del stem_tokens
              gc.collect()
               
              #Profile mem usage
              if not (self.files_read % 1000):
                for tt in self.tokenized_texts[start_idx:start_idx+1000]:
                  for tok in tt:
                    size+=sys.getsizeof(tok)
                print ('Tokenized texts:  {} ; Files read = {} '.format(size, self.files_read ))
                start_idx+=1000

            data_f.close() 
 
      print('Files successfully added to database: {}'.format(self.files_read)) 

      if self.files_read > 0:
        #Assign an integer to each unique word in the texts
        self.word_to_id = corpora.Dictionary(self.tokenized_texts)

        #Convert tokenized text into bow with id's used by gensim for LDA or LSI
        corpus = [self.word_to_id.doc2bow(text) for text in self.tokenized_texts]
        
        #Split into train and test corpus
        random_sample = random.sample( range(len(corpus)) , int(math.floor( len(corpus) * self.test_pct)))
        if random_sample:
          self.test_set = [text for idx, text in enumerate(corpus) if idx in random_sample]
          for idx,text in enumerate(corpus):
            if idx in random_sample:
              del corpus[idx]
          self.train_set = corpus
          print ('Training set size: {}, Test set size: {}'.format(len(self.train_set), len(self.test_set)))

        else:
          self.train_set = corpus
          print ('Training set size: {}, Test set size: {}'.format(len(self.train_set), 0))
   

        #Profile mem usage
        size = 0
        for text in self.train_set:
          for tup in text:
            size+=sys.getsizeof(tup)
        print ('Train_set:  {}'.format(size))

    else:
      print "Initialized empty database."
    

  def add_text_file(self, data_file):
    """Adds another .txt file to the database 
        Note: This is HIGHLY inefficient in time and space. 
        Args: 
          data_file (str): absolute or relative path to the .txt file
 
    """
     
    assert(os.path.isfile(data_file)), "Invalid file path"
    assert(data_file.endswith('.txt')), "Invalid file type"

    #Read in data
    with codecs.open(data_file, 'r', 'utf-8-sig') as data_f:
      doc = data_f.read().replace('\n', ' ')
      #Tokenize 
      tokens = word_tokenize(doc.lower())
      #Remove stop words
      stop_tokens = [token for token in tokens if token not in self.stop_words]
      #Step text using Porter Stemming Algorithm
      stem_tokens = [self.stemmer.stem(token) for token in stop_tokens]
      self.tokenized_texts.append(stem_tokens)
      self.files_read+=1

      #Clear up unused variables for efficient mem usage
      del doc
      del tokens
      del stop_tokens
      del stem_tokens
      gc.collect()

    data_f.close()  

    #Assign an integer to each unique word in the texts
    self.word_to_id = corpora.Dictionary(self.tokenized_texts)

    #Convert tokenized text into bow with id's used by LDA (or LSA)
    corpus = [self.word_to_id.doc2bow(text) for text in self.tokenized_texts]

    #Split into train and test corpus
    random_sample = random.sample( range(len(corpus)) , int(math.floor( len(corpus) * self.test_pct)))
      
    if random_sample: 
      self.test_set = [text for idx, text in enumerate(corpus) if idx in random_sample]
      for idx,text in enumerate(corpus):
        if idx in random_sample:
          del corpus[idx]
      self.train_set = corpus
      print ('Training set size: {}, Test set size: {}'.format(len(self.train_set), len(self.test_set)))
    else:
      self.train_set = corpus
      print ('Training set size: {}, Test set size: {}'.format(len(self.train_set), 0))

  def add_data_dir(self, data_dir):
    """Adds .txt files from data_dir to the database 
        Note: This is inefficient in time and space. 
        Args: 
          data_dir (str): absolute or relative path to the dir
                             containing .txt files
 
    """
     
    assert(os.path.isdir(data_dir)), "Invalid data directory path"

    for root, dirs, files in os.walk(data_dir):
      #Iterate over files
      for f in files:
        #Read in data for all .txt files
        if f.endswith('.txt'):
          with codecs.open(data_dir + '/' + f, 'r', 'utf-8-sig') as data_f:
            doc = data_f.read().replace('\n', ' ')
            #Tokenize 
            tokens = word_tokenize(doc.lower())
            #Remove stop words
            stop_tokens = [token for token in tokens if token not in self.stop_words]
            #Step text using Porter Stemming Algorithm
            stem_tokens = [self.stemmer.stem(token) for token in stop_tokens]
            self.tokenized_texts.append(stem_tokens)
            self.files_read+=1
            #Clear up unused variables for efficient mem usage
            del doc
            del tokens
            del stop_tokens
            del stem_tokens
            gc.collect()

          data_f.close()  

    #Assign an integer to each unique word in the texts
    self.word_to_id = corpora.Dictionary(self.tokenized_texts)

    #Convert tokenized text into bow with id's used by LDA (or LSA)
    corpus = [self.word_to_id.doc2bow(text) for text in self.tokenized_texts]

    #Split into train and test corpus
    random_sample = random.sample( range(len(corpus)) , int(math.floor( len(corpus) * self.test_pct)))
      
    if random_sample: 
      self.test_set = [text for idx, text in enumerate(corpus) if idx in random_sample]
      for idx,text in enumerate(corpus):
        if idx in random_sample:
          del corpus[idx]
      self.train_set = corpus
      print ('Training set size: {}, Test set size: {}'.format(len(self.train_set), len(self.test_set)))
    else:
      self.train_set = corpus
      print ('Training set size: {}, Test set size: {}'.format(len(self.train_set), 0))


  def store_to_disk(self, file_path):
    """Store the database object to disk for future use
    
     Args: 
      file_path(str): absolute or relative path of file to store the db in
    """
    
    assert(os.path.dirname(file_path)), 'Invalid directory provided to save file'
    assert(os.access(os.path.dirname(file_path), os.W_OK)), 'Need write permissions to parent dir'

    with open(file_path, 'w') as f:
      pickle.dump([self.train_set,
                   self.test_set, 
                   self.stop_words, 
                   self.stemmer, 
                   self.files_read, 
                   self.tokenized_texts, 
                   self.word_to_id,
                   self.train_epoch_idx,
                   self.batch_size, 
                   self.db_name],
                   f)

  def load_from_disk(self, file_path):
    """Load the corpus from disk 
     Args: 
      file_path(str): absolute or relative path of file to store the db in
    """
    
    assert(os.path.isfile(file_path)), 'Invalid file path to load db'

    with open(file_path) as f:
      self.train_set,\
      self.test_set,\
      self.stop_words,\
      self.stemmer,\
      self.files_read,\
      self.tokenized_texts,\
      self.word_to_id,\
      self.train_epoch_idx,\
      self.batch_size,\
      self.db_name = pickle.load(f)

  def prep_train_epoch(self, batch_size=1, num_epochs=1):
    """ Prepare the training corpus for one epoch by splitting the training
        corpus into mini batches of batch_size. This function needs to be
        called before every epoch of training as it does the book keeping
        required to send minibatches of data 

    Args: batch_size(int): Size of every mini batch that will be returned
                           by get_train_batch
          num_epochs(int): Number of epochs to train for

    """
    assert(self.train_set is not None),'There is no training data in the database'
    assert(batch_size > 0),'Batch size must be a positive int less than size of training set'
    assert(batch_size<=len(self.train_set)),'Batch size must be a positive int less than size of training set'
    assert(num_epochs > 0),'Num epochs must be a positive int'

    self.batch_size = batch_size
    self.train_epoch_idx = []

    for i in range(num_epochs):
      self.train_epoch_idx.extend(random.sample(range(len(self.train_set)), len(self.train_set)))

    if not self.train_epoch_idx:
      print "Warning: There is no training data in the database."

  def get_mini_batch(self):
    """Get a mini batch of data 
       Note that if less than batch_size of training samples are remaining, then we return the remaining samples.
    """

    assert(self.train_epoch_idx is not None),'Need to call prep_train_epoch(batch_size) before calling get_mini_batch()'
    if self.train_epoch_idx:
      if len(self.train_epoch_idx) >= self.batch_size:
        mb_idx = self.train_epoch_idx[0:self.batch_size]
        self.train_epoch_idx = self.train_epoch_idx[self.batch_size:len(self.train_epoch_idx)]
        mini_batch = [text for idx,text in enumerate(self.train_set) if idx in mb_idx]
        return mini_batch


      elif len(self.train_epoch_idx) < self.batch_size and len(self.train_epoch_idx) > 0:
        mb_idx = self.train_epoch_idx[:]
        self.train_epoch_idx = []
        mini_batch = [text for idx,text in enumerate(self.train_set) if idx in mb_idx]
        return mini_batch

    else:
      print "Training data exhausted."
      return []

  def get_train_set(self):
    """Get the test set. In case the test set is very large, get a fraction of the test set

    Args:
      test_pct(int): percet of test set to return 
    """

    if (self.train_set is None) or (not self.train_set):
      print "There is no train data in the database."
      return

    else:
      return self.train_set


  def get_test_set(self, test_pct=100):
    """Get the test set. In case the test set is very large, get a fraction of the test set

    Args:
      test_pct(int): percet of test set to return 
    """

    assert(test_pct >= 0 and test_pct <= 100),'test_pct must be a positive int <= 100'
    if (self.test_set is None) or (not self.test_set):
      print "There is no test data in the database."
      return

    if test_pct != 100:
      mb_idx = random.sample(range(len(self.test_set)), int(math.floor(len(self.train_set)*test_pct*0.01)) )
      mini_batch = [text for idx,text in enumerate(self.test_set) if idx in mb_idx]
      return mini_batch
    else:
      return self.test_set
    

  def get_word2id(self):
    assert(self.word_to_id is not None),'Database is empty. Please initialize database with data.'
    return self.word_to_id

  def corpus_size(self):
    return self.files_read

  def train_set_size(self):
    return len(self.train_set)

  def test_set_size(self):
    return len(self.test_set)

  def get_name(self):
    return self.db_name

