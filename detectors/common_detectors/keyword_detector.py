import numpy as np
# nlp modules
import spacy
from spacy.matcher import Matcher
from spacy.language import Language
from spacy.tokens import Token

Token.set_extension("is_silence_tag", default=False)

@Language.component("set_silence_tags")
def set_silence_tags(doc):
    for token in doc:
        if token.text == "[silence]":
            token._.is_silence_tag = True    
    return doc

class KeywordDetector:
    def __init__(self, keywords):
        self.keywords = keywords

        # Load spacy vocabulary
        self.nlp = spacy.load("en_core_web_md")
        self.nlp.tokenizer.add_special_case("[silence]", [{"ORTH": "[silence]"}])
        self.nlp.add_pipe("set_silence_tags", first=True)

        # Create spacy matcher
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
    