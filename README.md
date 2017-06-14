# LDA-Topic-Modeling
Latent Dirichlet Allocation for Topic Modeling

Python implementation of LDA Topic modeling using gensim. 
\textbf{Implementation} <br />
\tab We implement the model in Python and use the Gensim \cite{gensim} library for the LDA modeling. The library provides a simple API to trian and test the model. We provide a summary of the scripts used by the system: <br /> <br />
1. database.py: <br />
\tab This script provides the Database object. The Database object accepts a data directory to read the documents in from and abstracts away the process of: <br />
\tab \tab (a) Reading in multiples documents from a given directory <br />
\tab \tab (b) Tokenizing the text <br />
\tab \tab (c) Removing stop words from the text (for better training and inference) <br />
\tab \tab (d) Reducing words to their stems <br />
\tab Further, it splits the data into a training set and a test set. During training, it provides mini-batches of data. <br />

2. lda.py: <br />
\tab This script provides the LDA object. The LDA object accepts an instance of the Database object to train on. The LDA object abstracts away the process of: <br />
\tab \tab (a) Getting data from the database <br />
\tab \tab (b) Training on mini-batches of data <br />
\tab \tab (c) Saving and reloading trained models <br />
\tab \tab (d) Visualizing training and testing results <br />

3. main.py:<br />
This script creates a Database object for every directory that has .txt files in it. It then creates an LDA object that is trained on all the Database objects one by one. Finally, it saves the trained model. <br />

4. split\_data.sh:<br />
This is a shell script that takes in the data directory formed from the data \href{https://drive.google.com/file/d/0BwoY6ta4F2xzcmZGRXZDTVhBVGc/view?usp=sharing}{here}. It then splits the data into sub directories of 10,000 .txt files each. The directory containing these sub directories is then fed into main.py. main.py then creates a Database object for each of these sub directories (one at a time) and trains the LDA object. This script was written so as to train an LDA model on all the data without running out of memory. <br /> 
