from ..common_detectors.keyword_detector import KeywordDetector
# from .keywords_config import keywords

class FillerKeywordDetector(KeywordDetector):
    def __init__(self, nlp, keywords, flag_nonwords=False):
        '''
        Initializes the FillerKeywordDetector class.

        args:
            nlp (spacy.lang) - spaCy language model
            keywords (list) - list of keywords to detect
            flag_nonwords (bool) - flag nonwords as filler
        '''
        super().__init__(nlp, keywords)
        self.flag_nonwords = flag_nonwords

    def detect(self, text):
        '''
        Extends parent class detect method to also  
        '''
        output, doc = super().detect(text, return_doc=True)

        if self.flag_nonwords:
            output += [(token.idx, token.idx + len(token), token.text) for token in doc if token.is_oov and not (token.is_punct or token.is_space or token._.is_silence_tag or token._.is_inaudible_tag or token._.is_event_tag)]

        return output
