import numpy as np
from itertools import islice

class KeywordDetector:
    def __init__(self, keywords):
        self.keywords = keywords

    def get_keywords(self):
        '''
        Get the keywords.
        '''
        return self.keywords

    def ngrams(self, tokens, n):
        '''
        Generate n-grams.
        '''
        return list(zip(*(islice(tokens, i, None) for i in range(n))))

    def detect(self, text):
        '''
        Detect keywords in text.
        '''
        tokens = text.split()    # tokenize text
        
        output = np.zeros(len(tokens))
        for kw in self.keywords:
            kw_len = len(kw.split())
            n_grams = self.ngrams(tokens, kw_len)     # generate n-grams the same length as the keyword
            for i, ng in enumerate(n_grams):
                phrase = " ".join([tok.lower() for tok in ng])
                if phrase == kw:    # check for exact match
                    indices = list(range(i, i + kw_len))
                    output[indices] = 1

        return output