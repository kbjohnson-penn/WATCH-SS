import re
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
    for tag_name in ["is_silence_tag", "is_inaudible_tag", "is_event_tag"]:
        if not Token.has_extension(tag_name):
            Token.set_extension(tag_name, default=False)

    # Custom pipeline component to set transcript special tags
    @Language.component("set_transcript_tags")
    def set_transcript_tags_component(doc):
        for token in doc:
            # Handle silence tags
            if re.match(r'\[silence\b.*\]', token.text):
                token._.is_silence_tag = True
            # Handle inaudible tags
            elif token.text == "[inaudible]":
                token._.is_inaudible_tag = True
            # Handle all other event tags
            elif re.match(r"^\[.*\]$", token.text):
                token._.is_event_tag = True
        return doc    

    # Load the spaCy model
    nlp = spacy.load(model)

    # Customize the tokenizer to look for text brackets
    nlp.tokenizer = Tokenizer(nlp.vocab, 
                              rules=nlp.Defaults.tokenizer_exceptions,
                              prefix_search=spacy.util.compile_prefix_regex(nlp.Defaults.prefixes).search,
                              suffix_search=spacy.util.compile_suffix_regex(nlp.Defaults.suffixes).search,
                              infix_finditer=spacy.util.compile_infix_regex(nlp.Defaults.infixes).finditer,
                              token_match=re.compile(r"\[[^\]]+\]").match
    )
    
    # Add custom tagging component to pipeline
    nlp.add_pipe("set_transcript_tags", first=True)
    
    print(f"Updated pipeline: {nlp.pipe_names}")
    return nlp



@mlflow.trace
def llm_call(client, model: str, dev_prompt: str, usr_prompt: str, response_fmt: dict = {"type": "text"}, temperature: float = 0.0, top_p: float = 0.95):
    '''
    LLM call function with MLFlow tracing active.

    args:
        client (openai.OpenAI) : LLM client
        model (str) : LLM endpoint name
        dev_prompt (str) : developer prompt
        usr_prompt (str) : user prompt
        response_fmt (dict or None) : response format, default = {"type": "text"}
        temperature (float) : temperature parameter, default = 0.0
        top_p (float) : top_p parameter, default = 0.95

    return:
        (str or dict) : LLM response as string or JSON dict if response_fmt indicates JSON
    '''
    messages = []
    if dev_prompt is not None:
        messgaes.append({"role": "developer", "content": dev_prompt})
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
