from spacy.matcher import Matcher

class KeywordDetector:
    def __init__(self, keywords, nlp):
        '''
        Initialize the keyword detector.

        args:
            keywords (list) - list of keywords to detect
            nlp (spacy.lang) - spaCy language model
        '''
        self.keywords = keywords
        self.nlp = nlp

        # Create spaCy matcher
        self.matcher = Matcher(self.nlp.vocab)
        patterns = [[{"LOWER": w}  for w in kw.lower().split()] for kw in self.keywords]
        self.matcher.add("fillers", patterns)

    def get_keywords(self):
        '''
        Get the keywords.
        '''
        return self.keywords

    def detect(self, text, return_doc=False):
        '''
        Detect keywords in text.

        args:
            text (str) - input text
            return_doc (bool) - whether to return the spacy doc (default is False)
        '''
        # Tokenize the input text
        doc = self.nlp(text)

        # Run spacy matcher
        matches = self.matcher(doc)

        output = []
        for match_id, start_token, end_token in matches:
            span = doc[start_token:end_token]
            output.append((span.start_char, span.end_char))

        return output, doc if return_doc else output
    