# NLP_trigram_model

The Trigram Language Model is implemented in python. 
The implementation is based on a template provided by Prof. Daniel Bauer. 

# Handling missing word: 
The way to deal with unseen words is to use a pre-defined lexicon before we extract ngrams. The function corpus_reader has an optional parameter lexicon, which should be a Python set containing a list of tokens in the lexicon. All tokens that are not in the lexicon will be replaced with a special "UNK" token.

# Smoothing method 
Using linear interpolation between the raw trigram, unigram, and bigram probabilities to smooth probabilities. 

![alt text](https://github.com/lt616/NLP_trigram_model/blob/master/interpolation.png) 

In this project we set lambda1 == lambda2 == lambda3 == 1/3 

# Performance benchmark 
* Compute Perplexity based on brown_train.txt (train file) and brown_test.txt (test file). 

The perlexity is 300.17653468276893 

* Compute correct prediction rate based on ets_toefl_data dataset. 

The accuracy is 84.86% 





