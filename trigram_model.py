import sys
from collections import defaultdict
import math
import random
import os
import os.path 
import glob
"""
COMS W4705 - Natural Language Processing - Fall 2018
Homework 1 - Programming Component: Trigram Language Models
Daniel Bauer
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n < len(sequence).
    """ 
    results = [] 

    # Check if n valid 
    count_start = 1 
    if not n > 0: 
        return results 
    elif n > 1: 
        count_start = n - 1 

    # Append "START" & "STOP" keywords to the sequence 
    ext_sequence = [] 
    for i in range(0, count_start): 
        ext_sequence += ["START"] 
    ext_sequence += sequence + ["STOP"] 

    for i in range(len(ext_sequence) - n + 1): 
        result = [] 
        # print(i) 
        for j in range(i, i + n): 
            # print(j - i) 
            result.append(ext_sequence[j]) 
        results.append(tuple(result)) 

    return results 


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator) 

    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
        self.unigramcounts = {} # might want to use defaultdict or Counter instead
        self.bigramcounts = {} 
        self.trigramcounts = {} 
        self.totalwordcounts = 0  

        for sequence in corpus: 
            for token in get_ngrams(sequence, 1): 
                if not token in self.unigramcounts: 
                    self.unigramcounts[token] = 0 
                self.unigramcounts[token] += 1 

            for token in get_ngrams(sequence, 2): 
                if not token in self.bigramcounts: 
                    self.bigramcounts[token] = 0 
                self.bigramcounts[token] += 1 

            for token in get_ngrams(sequence, 3): 
                if not token in self.trigramcounts: 
                    self.trigramcounts[token] = 0 
                self.trigramcounts[token] += 1 

            # Calculate total number of words 
            self.totalwordcounts += len(sequence)  

        # Set start word in bigram collection for trigram 
        self.bigramcounts[("START", "START")] = self.unigramcounts[("START",)] 

        return

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """         

        temp = [] 
        temp.append(trigram[0]) 
        temp.append(trigram[1]) 

        # If a word does not appear in lexicon then set it to UNK 
        temp = self.filterUNK(tuple(temp)) 
        trigram = self.filterUNK(tuple(trigram)) 

        try: 
            return float(self.trigramcounts[trigram]) / self.bigramcounts[temp] 
        except: 
            return float(0) 


    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """ 

        temp = [] 
        temp.append(bigram[0]) 

        # If a word does not appear in lexicon then set it to UNK 
        temp = self.filterUNK(tuple(temp)) 
        bigram = self.filterUNK(bigram) 

        try: 
            return float(self.bigramcounts[bigram]) / self.unigramcounts[tuple(temp)] 
        except: 
            return float(0) 

    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """ 

        # If a word does not appear in lexicon then set it to UNK 
        unigram = self.filterUNK(unigram) 

        try: 
            return float(self.unigramcounts[unigram]) / self.totalwordcounts 
        except: 
            return float(0) 

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.  
        # return 0.0

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        return result            

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0 

        temp_unigram = [] 
        temp_unigram.append(trigram[2]) 
        p_unigram = self.raw_unigram_probability(tuple(temp_unigram)) 

        temp_bigram = [] 
        temp_bigram.append(trigram[1]) 
        temp_bigram.append(trigram[2]) 
        p_bigram = self.raw_bigram_probability(tuple(temp_bigram)) 

        p_trigram = self.raw_trigram_probability(trigram) 

        return lambda1 * p_trigram + lambda2 * p_bigram + lambda3 * p_unigram 
 
        return 0.0
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """ 

        p_log = 0 
        for trigram in get_ngrams(sentence, 3): 
            p_log += math.log2(self.smoothed_trigram_probability(trigram)) 

        return p_log 

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """ 

        sum = 0 
        count = 0 
        for sentence in corpus: 
            sum += self.sentence_logprob(sentence) 
            count += len(sentence) 

        sum /= count 

        return pow(2, -sum) 


    def filterUNK(self, tp): 

        temp = []
        for i in range(0, len(tp)): 
            if not tp[i] in self.lexicon: 
                temp.append("UNK") 
            else: 
                temp.append(tp[i]) 

        return tuple(temp) 


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

    # Essay scoring experiment: 
    high_model_file = training_file1  
    low_model_file = training_file2 
    high_model = TrigramModel(high_model_file) 
    low_model = TrigramModel(low_model_file) 

    correct_count = 0 
    count = 0 
    for test_file in glob.glob(testdir1): 
        high_corpus = corpus_reader(test_file, high_model.lexicon) 
        high_pp = high_model.perplexity(high_corpus) 

        low_corpus = corpus_reader(test_file, low_model.lexicon) 
        low_pp = low_model.perplexity(low_corpus) 

        count += 1 

        if high_pp < low_pp: 
            correct_count += 1 

    for test_file in glob.glob(testdir2): 
        high_corpus = corpus_reader(test_file, high_model.lexicon) 
        high_pp = high_model.perplexity(high_corpus) 

        low_corpus = corpus_reader(test_file, low_model.lexicon) 
        low_pp = low_model.perplexity(low_corpus) 

        count += 1 

        if low_pp < high_pp: 
            correct_count += 1 
        
    return float(correct_count) / count  





if __name__ == "__main__":

    # model = TrigramModel(sys.argv[1]) 
    model = TrigramModel(sys.argv[1]) 
    # "/Users/CherryZHAO/Desktop/18Fall/COMS4705/assignment01/hw1_data/brown_train.txt" 
    # "/Users/CherryZHAO/Desktop/18Fall/COMS4705/assignment01/hw1_data/brown_test.txt" 
    # model = TrigramModel(model_file) 

    # Test for part 1 
    # print(get_ngrams(["natural","language","processing"],1))
    # print(get_ngrams(["natural","language","processing"],2)) 
    # print(get_ngrams(["natural","language","processing"],3)) 

    # Test for part 2 
    # print(model.trigramcounts[('START','START','the')]) 
    # print(model.bigramcounts[('UNK','UNK')]) 
    # print(model.unigramcounts[('the',)]) 
    
    # print(model.totalwordcounts) 

    # print(model.raw_unigram_probability(('the',))) 
    # print(model.raw_bigram_probability(('the','sands'))) 
    # print(model.raw_bigram_probability(('sands','of'))) 

    # print(model.raw_trigram_probability(('the', 'sands', 'of'))) 

    # print(model.smoothed_trigram_probability(('the', 'sands', 'of'))) 

    # print(model.sentence_logprob(['the', 'fulton', 'county', 'grand', 'jury', 'said', 'friday', 'an', 'investigation', 'of', 'atlanta', "'s", 'recent', 'primary', 'election', 'produced', '``', 'no', 'evidence', "''", 'that', 'any', 'irregularities', 'took', 'place', '.']))

    # Testing perplexity: 
    dev_corpus = corpus_reader(sys.argv[2], model.lexicon) 
    pp = model.perplexity(dev_corpus)
    print(pp) 


    # Test for func essay_scoring_experiment 
    high_model_file = "/Users/CherryZHAO/Desktop/18Fall/COMS4705/assignment01/hw1_data/ets_toefl_data/train_high.txt" 
    low_model_file = "/Users/CherryZHAO/Desktop/18Fall/COMS4705/assignment01/hw1_data/ets_toefl_data/train_low.txt" 

    test_high_dir = "/Users/CherryZHAO/Desktop/18Fall/COMS4705/assignment01/hw1_data/ets_toefl_data/test_high/*.txt" 
    test_low_dir = "/Users/CherryZHAO/Desktop/18Fall/COMS4705/assignment01/hw1_data/ets_toefl_data/test_low/*.txt" 

    acc = essay_scoring_experiment(high_model_file, low_model_file, test_high_dir, test_low_dir)  
    print(acc) 

