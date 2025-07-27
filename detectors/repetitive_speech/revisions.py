# similarity modules
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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