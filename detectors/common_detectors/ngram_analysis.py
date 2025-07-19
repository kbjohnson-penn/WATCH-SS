import numpy as np
from itertools import islice

class NgramAnalysis:
    def __init__(self, nlp, max_N=1, window_size=5):
        '''
        Initialize the n-gram analysis detector.

        args:
            nlp (spacy.lang) - spaCy language model
            max_N (int) : maximum N-gram size (default = 1)
            window_size (int) : search window size for comparing n-grams as number of previous tokens (default = 5)
        '''
        self.nlp = nlp
        self.max_N = max_N
        self.window_size = window_size
    
    def _ngrams(self, tokens, n):
        '''
        Generate n-grams.

        args:
            tokens (list<spacy.Token>) : spacy document
            n (int) : n-gram size

        return:
            (list<tuple<spacy.Token>>) list of n-grams
        '''
        assert(n > 0)
        return list(zip(*(islice(tokens, i, None) for i in range(n))))
    
    def _compare_ngrams(self, ng1, ng2, doc):
        '''
        Compare two n-grams and determine if they are similar.

        args:
            ng1 (tuple<spacy.Token>) : first n-gram
            ng2 (tuple<spacy.Token>) : second n-gram
            doc (spacy.Doc) : spacy document

        return:
            (bool) : True if similar otherwise False
        '''
        return NotImplementedError
    
    def detect(self, text):
        '''
        Detect repeated n-grams in the input text.
        
        args:
            text (str) : input text

        return:
            (dict) : JSON object with key "detections" that a list of repetition
            objects, each having the following keys:
                - "type" : "repetition"
                - "phrase1" : first phrase
                - "span1" : list of character span of first phrase
                - "phrase2" : second phrase
                - "span2" : list of character span of second phrase
        '''
        # Tokenize the input text
        doc = self.nlp(text)
        # Filter out punctuation and whitespace tokens
        tokens = [token for token in doc if not (token.is_punct or token.is_space or token._.is_silence_tag or token._.is_inaudible_tag or token._.is_event_tag)]

        output = {"detections": []}
        for n in range(1, self.max_N+1):
            ngs = self._ngrams(tokens, n)

            for i in range(len(ngs)):
                # Compare current n-gram to preceeding, non-overlapping n-grams
                for j in range(max(0, i - self.window_size), i-n+1):
                    if self._compare_ngrams(ngs[j], ngs[i], doc):
                        output["detections"].append({
                                "type": "repetition",
                                "phrase1": doc[ngs[j][0].i:ngs[j][-1].i+1].text, 
                                "span1": [ngs[j][0].idx, ngs[j][-1].idx + len(ngs[j][-1])], 
                                "phrase2": doc[ngs[i][0].i:ngs[i][-1].i+1].text, 
                                "span2": [ngs[i][0].idx, ngs[i][-1].idx + len(ngs[i][-1])]
                        })

        return output
