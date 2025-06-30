# import mlflow
from ..common_detectors.keyword_detector import KeywordDetector
# from .keywords_config import keywords

class FillerKeywordDetector(KeywordDetector):
    def __init__(self, keywords, nlp, flag_nonwords=False):
        '''
        Initializes the FillerKeywordDetector class.

        args:
            keywords (list) - list of keywords to detect
            nlp (spacy.lang) - spaCy language model
            flag_nonwords (bool) - flag nonwords as filler
        '''
        super().__init__(keywords, nlp)
        self.flag_nonwords = flag_nonwords

    # @mlflow.trace
    def detect(self, text):
        '''
        Extends parent class detect method to also  
        '''
        output, doc = super().detect(text, return_doc=True)

        if self.flag_nonwords:
            output += [(token.idx, token.idx + len(token)) for token in doc if token.is_oov and not (token.is_punct or token.is_space or token._.is_silence_tag)]

        return output
