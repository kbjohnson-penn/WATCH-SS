from spacy.matcher import Matcher

class KeywordDetector:
    def __init__(self, nlp, keywords):
        '''
        Initialize the keyword detector.

        args:
            nlp (spacy.lang) - spaCy language model
            keywords (list) - list of keywords to detect
        '''
        self.nlp = nlp
        self.keywords = keywords

        # Create spaCy matcher
        self.matcher = Matcher(self.nlp.vocab)
        patterns = [[{"LOWER": w}  for w in kw.lower().split()] for kw in self.keywords]
        self.matcher.add("keywords", patterns)

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

        return:
            (dict) : JSON object with key "detections" that a list of filler
            objects, each having the following keys:
                - "text" : the detected filler
                - "span" : character span of detected filler
        '''
        # Tokenize the input text
        doc = self.nlp(text)

        # Run spacy matcher
        matches = self.matcher(doc)

        output = {"detections": []}
        for match_id, start_token, end_token in matches:
            span = doc[start_token:end_token]
            output["detections"].append({"text": span.text, "span": [span.start_char, span.end_char]})

        return output, doc if return_doc else output
    