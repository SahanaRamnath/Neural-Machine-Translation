# Neural-Machine-Translation
Implementing NMT on an IWSLT dataset, using sequence-to-sequence (seq2seq) models, in Tensorflow

# Dependencies
* Libraries tensorflow and nltk in python 2.x

# Usage
Code to be run in terminal to train and evaluate the model : 
`python main_nmt.py`

# Codes
* **main_nmt.py**
* **data_preparation.py** : Data Extraction and Preprocessing
* **parameters.py** : Initialising all the parameters and hyperparamers used.
* **model_attention.py** : Class that defines the structure of the encoder-decoder model.
* **train_nmt.py** : Creates train,test and validation instances. Trains and evaluates the model.
* **basic_functions.py** : Function definitions for basic architecture and evaluation.
* **additional_functions.py** : Function definitions to load data and format the output.
* **calculate\_bleu\_score.py** : Calculates bleu score. Usage : `python calculate_bleu_score.py /path/to/reference_file /path/to/predicted_file`

# Dataset Used
English-Vietnamese parallel corpus of TED Talks, provided by the [IWSLT Evaluation Campaign](https://sites.google.com/site/iwsltevaluation2015/).
Preprocessed data from [The Stanford NLP group](https://nlp.stanford.edu/projects/nmt/) was used to train and test the models.
* train.en
* train.vi
* tst2013.en (test set)
* tst2013.vi (test set)
* tst2012.en (validation set)
* tst2012.vi (validation set)
* vocab.en
* vocab.vi

# References
* [Neural Machine Translation(seq2seq) Tutorial](https://www.tensorflow.org/tutorials/seq2seq) by Tensorflow and their [source code](https://github.com/tensorflow/nmt) 