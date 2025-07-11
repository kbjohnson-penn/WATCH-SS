import numpy as np
from itertools import islice
# similarity modules
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class RepetitionNgramAnalysis:
    def __init__(self, nlp, max_n=5, max_lookback_tokens=5, comparator="exact", sim_threshold=None):
        '''
        Initialize the repetition n-gram analysis detector.

        args:
            nlp (spacy.lang) - spaCy language model
            max_n (int) : maximum n-gram size (default = 5)
            max_lookback_tokens (int) : maximum number of tokens to look back for comparison (default = 10)
            comparator (str) : n-gram comparison method. options include "exact", "lemma_exact", "SBERT_sim", and "fuzzy". (default = "exact")
            sim_threshold (float or None) : similarity threshold (default = None)
        '''
        self.nlp = nlp
        self.max_n = max_n
        self.max_lookback_tokens = max_lookback_tokens
        self.sim_threshold = sim_threshold

        if comparator == "exact":
            self.compare_func = self._compare_ngrams_exact
        elif comparator == "lemma_exact":
            self.compare_func = self._compare_ngrams_lemma_exact
        elif comparator == "spaCy_sim":
            self.compare_func = self._compare_ngrams_spaCy_sim
        elif comparator == "SBERT_sim":
            self.compare_func = self._compare_ngrams_SBERT_sim
            self.sbert = SentenceTransformer("all-MiniLM-L6-v2")
        elif comparator == "fuzzy":
            self.compare_func = self._compare_ngrams_fuzzy
        else:
            raise ValueError(f"Invalid comparator: {comparator}")
    
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

    def _compare_ngrams_exact(self, ng1, ng2, doc=None, sim_threshold=None):
        '''
        Compare two n-grams by comparing the text verbatim.

        args:
            ng1 (tuple<spacy.Token>) : first n-gram
            ng2 (tuple<spacy.Token>) : second n-gram
            doc (spacy.Doc) : spacy document (default = None)
            sim_threshold (float or None) : similarity threshold (default = None)

        return:
            (bool) True if same n-grams, False otherwise
        '''
        assert(len(ng1) == len(ng2))
        return all([ng1[i].lower_ == ng2[i].lower_ for i in range(len(ng1))])

    def _compare_ngrams_lemma_exact(self, ng1, ng2, doc=None, sim_threshold=None):
        '''
        Compare two n-grams by comparing the lemma's of the words.

        args:
            ng1 (tuple<spacy.Token>) : first n-gram
            ng2 (tuple<spacy.Token>) : second n-gram
            doc (spacy.Doc) : spacy document (default = None)
            sim_threshold (float or None) : similarity threshold (default = None)

        return:
            (bool) True if same n-grams, False otherwise
        '''
        assert(len(ng1) == len(ng2))
        return all([ng1[i].lemma_ == ng2[i].lemma_ for i in range(len(ng1))])

    def _compare_ngrams_spaCy_sim(self, ng1, ng2, doc=None, sim_threshold=0.9):
        '''
        Compare two n-grams using spaCy embeddings and cosine similarity.

        args:
            ng1 (tuple<spacy.Token>) : first n-gram
            ng2 (tuple<spacy.Token>) : second n-gram
            doc (spacy.Doc) : spacy document (default = None)
            sim_threshold (float) : similarity threshold (default = 0.9)

        return:
            (bool) True if same n-grams, False otherwise
        '''
        assert(len(ng1) == len(ng2))
        embdg_ng1 = np.mean([token.vector for token in ng1], axis=0)   # TODO: consider the case where a token is OOV and has no vector
        embdg_ng2 = np.mean([token.vector for token in ng2], axis=0)
        sim = cosine_similarity(embdg_ng1.reshape(1, -1), embdg_ng2.reshape(1, -1))[0,0]
        # print(sim)
        return sim >= sim_threshold

    def _compare_ngrams_SBERT_sim(self, ng1, ng2, doc, sim_threshold=0.9):
        '''
        Compare two n-grams using Sentence BERT.

        args:
            ng1 (tuple<spacy.Token>) : first n-gram
            ng2 (tuple<spacy.Token>) : second n-gram
            doc (spacy.Doc) : spacy document
            sim_threshold (float) : similarity threshold (default = 0.9)

        return:
            (bool) True if same n-grams, False otherwise
        '''
        assert(len(ng1) == len(ng2))
        embdgs = self.sbert.encode([
            doc[ng1[0].i:ng1[-1].i+1].text,
            doc[ng2[0].i:ng2[-1].i+1].text,
        ])
        sim = self.sbert.similarity(embdgs, embdgs)[0,1].item()
        # print(sim)
        return sim >= sim_threshold        
    
    def _compare_ngrams_fuzzy(self, ng1, ng2, doc, sim_threshold=0.9):
        '''
        Compare two n-grams using fuzzy matching (Levenshtein distance).

        args:
            ng1 (tuple<spacy.Token>) : first n-gram
            ng2 (tuple<spacy.Token>) : second n-gram
            doc (spacy.Doc) : spacy document
            sim_threshold (float) : similarity threshold (default = 0.9)

        return:
            (bool) True if same n-grams, False otherwise
        '''
        assert(len(ng1) == len(ng2))
        str_ng1 = doc[ng1[0].i:ng1[-1].i+1].text
        str_ng2 = doc[ng2[0].i:ng2[-1].i+1].text
        sim = fuzz.ratio(str_ng1, str_ng2)
        # print(sim)
        return sim >= 100 * sim_threshold
    
    def _filter_redundant_repetitons(self, repetitions):
        '''
        Removes redundant repetitions from a list.

        args:
            repetitions (list<tuple[tuple[int, int], tuple[int, int]]>) : list of repetitions

        return:
            (list<tuple[tuple[int, int], tuple[int, int]]>) list of merged spans
        '''
        # Sort the spans by start index
        sorted_reps = sorted(repetitions, key=lambda x: (x[1][0], -x[1][1]))

        # Filter 
        filtered_reps = []
        filtered_reps.append(sorted_reps[0])

        for curr_rep in sorted_reps[1:]:
            last_kept_rep_span = filtered_results[-1][1]
            curr_rep_span = curr_rep[1]

            if (curr_rep_span[0] >= last_kept_rep_span[0]) and (curr_rep_span[1] <= last_kept_rep_span[1]):
                continue
            else:
                filtered_results.append(current_pair)

        return filtered_reps
    
    def detect(self, text, filter_redundant=False):
        '''
        Detect repetitive n-grams in the input text.
        
        args:
            text (str) : input text
            filter_redundant (bool) : whether to filter redundant repetitions (default = True)

        return:
            (list<tuple>) list of spans for repeated content
        '''
        # Tokenize the input text
        doc = self.nlp(text)
        # Filter out punctuation and whitespace tokens
        tokens = [token for token in doc if not (token.is_punct or token.is_space or token._.is_silence_tag)]

        repetitions = []
        for n in range(1, self.max_n+1):
            ngs = self._ngrams(tokens, n)

            for i in range(len(ngs)):
                # Compare current n-gram to preceeding, non-overlapping n-grams
                for j in range(max(0, i - self.max_lookback_tokens), i-n+1):
                    # print(ngs[j], ngs[i])
                    if self.compare_func(ngs[j], ngs[i], doc, self.sim_threshold):  
                        repetitions.append((
                            (ngs[j][0].idx, ngs[j][-1].idx + len(ngs[j][-1])), 
                            (ngs[i][0].idx, ngs[i][-1].idx + len(ngs[i][-1]))
                        ))

        if filter_redundant:
            repetitions = self._filter_redundant_repetitons(repetitions)

        return repetitions
