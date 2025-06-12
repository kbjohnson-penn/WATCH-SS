#!/bin/bash

echo "!!!!!!!!!!!!!!!RUNNING INIT BASH SCRIPT!!!!!!!!!!!!"
python -m pip install --upgrade spacy
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
python -m spacy download en_core_web_lg
