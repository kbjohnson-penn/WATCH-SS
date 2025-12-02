import re
import json
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score
# spaCy
import spacy
from spacy.language import Language
from spacy.tokens import Token
from spacy.tokenizer import Tokenizer
# MLFlow
import mlflow

llms = {
    "openai_gpt_4o": {
        "model": "gpt-4o",
        "tpm": 1e6,
        "rpm": 6.15e3
    },
    "openai_gpt35_turbo_instruct": {
        "model": "gpt-3.5-turbo",
        "tpm": 6e4,
        "rpm": 360
    },
    "meta_llama_v3_2_3b_instruct": {
        "model": "",
        "tpm": 6.54e5,
        "rpm": None
    },
    "meta_llama_v3_1_8b_instruct": {
        "model": "",
        "tpm": 5.7e5,
        "rpm": None
    }
}


def create_custom_nlp(model="en_core_web_md"):
    '''
    Load spaCy model and customize with special tokens
    for transcripts. 

    args:
        model (str): spaCy mode name

    return:
        (spacy.lang): spaCy model
    '''
    # Register custom attributes
    for tag_name in ["is_silence_tag", "is_inaudible_tag", "is_event_tag", "is_filler"]:
        if not Token.has_extension(tag_name):
            Token.set_extension(tag_name, default=False)

    # Customize the tokenizer to look for text brackets and & prefix
    SPECIAL_PATTERN = re.compile(r"(\[[^\]]+\]|&[a-zA-Z_]+)")

    @Language.component("merge_custom_tokens")
    def merge_custom_tokens_component(doc):
        with doc.retokenize() as retokenizer:
            for match in SPECIAL_PATTERN.finditer(doc.text):
                # Get the span of the match in the doc
                start, end = match.span()
                span = doc.char_span(start, end)
                # If a valid span is found (it aligns with token boundaries)
                if span is not None:
                    retokenizer.merge(span)
        return doc

    # Custom pipeline component to set transcript special tags
    @Language.component("set_transcript_tags")
    def set_transcript_tags_component(doc):
        for token in doc:
            # Handle silence tags
            if re.fullmatch(r'\[silence( \d+s)?\]', token.text):
                token._.is_silence_tag = True
            # Handle inaudible tags
            elif token.text == "[inaudible]":
                token._.is_inaudible_tag = True
            # Handle filler words
            elif re.fullmatch(r"&[a-zA-Z_]+", token.text):
                token._.is_filler = True
            # Handle all other event tags
            elif re.fullmatch(r"\[[^\]]+\]", token.text):
                token._.is_event_tag = True

        return doc

    # Load the spaCy model
    nlp = spacy.load(model)
    # Add custom merger component to pipeline
    nlp.add_pipe("merge_custom_tokens", first=True)
    # Add custom tagging component to pipeline
    nlp.add_pipe("set_transcript_tags", after="merge_custom_tokens")
    print(f"Updated pipeline: {nlp.pipe_names}")
    return nlp


def doc_word_count(doc):
    '''
    Function to count words in a spaCy doc object
    using custom nlp above.

    args:
        doc (spacy.tokens.doc.Doc): spaCy doc object

    return:
        (int): number of words in doc
    '''
    return sum([1 for token in doc if not (token.is_punct or token.is_space or token._.is_silence_tag or token._.is_inaudible_tag or token._.is_event_tag)])
    

@mlflow.trace
def llm_call(client, model: str, dev_prompt: str, usr_prompt: str, response_fmt: dict = {"type": "text"}, temperature: float = 0.0, top_p: float = 1.0):
    '''
    LLM call function with MLFlow tracing active.

    args:
        client (openai.OpenAI) : LLM client
        model (str) : LLM endpoint name
        dev_prompt (str) : developer prompt
        usr_prompt (str) : user prompt
        response_fmt (dict or None) : response format, default = {"type": "text"}
        temperature (float) : temperature parameter, default = 0.0
        top_p (float) : top_p parameter, default = 1.0

    return:
        (str or dict) : LLM response as string or JSON dict if response_fmt indicates JSON
    '''
    messages = []
    if dev_prompt is not None:
        messages.append({"role": "developer", "content": dev_prompt})
    if usr_prompt is not None:
        messages.append({"role": "user", "content": usr_prompt})
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format=response_fmt,
        temperature=temperature,
        top_p=top_p
    )

    if "json" in response_fmt["type"].lower():
        return json.loads(response.choices[0].message.content)
    else:
        return response.choices[0].message.content

def evaluate(true, pred, return_latex=False):
    prec = precision_score(true, pred)
    rec  = recall_score(true, pred)
    f1   = f1_score(true, pred)
    acc  = accuracy_score(true, pred)
    bacc = balanced_accuracy_score(true, pred)

    if return_latex:
        return f"{prec:.3f} & {rec:.3f} & {f1:.3f} & {acc:.3f} & {bacc:.3f} \\\\"

    return {"precision": prec, "recall": rec, "f1": f1, "accuracy": acc, "balanced_accuracy": bacc}
