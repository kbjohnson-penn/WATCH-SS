from ..common_detectors.keyword_detector import KeywordDetector
from .keywords_config import keywords as default_keywords

class VagueKeywordDetector(KeywordDetector):
    def __init__(self, nlp, keywords=default_keywords):
        '''
        Initializes the FillerKeywordDetector class.

        args:
            nlp (spacy.lang) - spaCy language model
            keywords (list) - list of keywords to detect
        '''
        super().__init__(nlp, keywords)
