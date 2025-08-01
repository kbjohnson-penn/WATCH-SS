from ..common_detectors.ngram_analysis import NgramAnalysis

class UnigramAnalysisDetector(NgramAnalysis):
    def __init__(self, nlp, window_size=2, comparator="exact"):
        '''
        Initialize the detector.

        args:
            nlp (spacy.lang) : spacy language model
            window_size (int) : window size for checking previous unigrams, default = 2
            comparator (str) : comparison function for unigrams, default = "exact" match
        '''
        self.comparator = comparator
        super().__init__(nlp, max_N=1, window_size=window_size)

    def _compare_ngrams_exact(self, ng1, ng2, doc=None):
        '''
        Compare two n-grams by comparing the text verbatim.

        args:
            ng1 (tuple<spacy.Token>) : first n-gram
            ng2 (tuple<spacy.Token>) : second n-gram
            doc (spacy.Doc) : spacy document (default = None)

        return:
            (bool) True if same n-grams, False otherwise
        '''
        assert(len(ng1) == len(ng2))
        return all([ng1[i].lower_ == ng2[i].lower_ for i in range(len(ng1))])

    def _compare_ngrams_lemma_exact(self, ng1, ng2, doc=None):
        '''
        Compare two n-grams by comparing the lemma's of the words.

        args:
            ng1 (tuple<spacy.Token>) : first n-gram
            ng2 (tuple<spacy.Token>) : second n-gram
            doc (spacy.Doc) : IGNORED

        return:
            (bool) True if same n-grams, False otherwise
        '''
        assert(len(ng1) == len(ng2))
        return all([ng1[i].lemma_ == ng2[i].lemma_ for i in range(len(ng1))])

    def _compare_ngrams(self, ng1, ng2, doc):
        if self.comparator == "exact":
            return self._compare_ngrams_exact(ng1, ng2, doc)
        elif self.comparator == "lemma_exact":
            return self._compare_ngrams_lemma_exact(ng1, ng2, doc)
        