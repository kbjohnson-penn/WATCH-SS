from ..common_detectors.keyword_detector import KeywordDetector
# from .keywords_config import keywords

class FillerKeywordDetector(KeywordDetector):
    def __init__(self, keywords, flag_nonwords=False):
        super().__init__(keywords=keywords)
        self.flag_nonwords = flag_nonwords

    def detect(self, text):
        '''
        Extends parent class detect method to also  
        '''
        output, doc = super().detect(text, return_doc=True)

        if self.flag_nonwords:
            output += [(token.idx, token.idx + len(token)) for token in doc if token.is_oov and len(token) > 1 and not token._.is_silence_tag]

        return output
