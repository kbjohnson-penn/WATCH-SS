from ..common_detectors.keyword_detector import KeywordDetector
# from .keywords_config import keywords

class FillerKeywordDetector(KeywordDetector):
    def __init__(self, keywords):
        super().__init__(keywords=keywords)
