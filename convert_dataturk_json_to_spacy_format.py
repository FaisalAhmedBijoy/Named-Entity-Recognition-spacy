import json
import re
import math
import spacy
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from spacy.tokens import DocBin
from spacy.util import filter_spans

from data_preprocessing import trim_entity_spans,preprocess_dataturks_json,train_test_split
def get_spacy_doc(file, data):
    nlp = spacy.blank("en")
    db = DocBin()

    for text, annot in tqdm(data):
        doc = nlp.make_doc(text)
        annot = annot['entities']
        ents = []
        entity_indices = []
        for start, end, label in annot:
            skip_entity = False
            for idx in range(start, end):
                if idx in entity_indices:
                    skip_entity = True
                    break
            if skip_entity == True:
                continue
            entity_indices = entity_indices + list(range(start, end))

            try:
                span = doc.char_span(start, end, label=label, alignment_mode="strict")
            except:
                continue

            if span is None:
                err_data = str([start, end]) + " " + str(text) + "\n"
                file.write(err_data)

            else:
                ents.append(span)
        try:
            doc.ents = ents
            db.add(doc)
        except:
            pass
    return db
def convert_dataturk_json_to_spacy_data_format(train_data, test_data, spacy_dir):
    print('dataturk to spacy function db')
    file = open("sample.txt", "w", encoding="utf-8")

    db = get_spacy_doc(file, train_data)
    train_spacy_path = spacy_dir + '/train_data.spacy'
    db.to_disk(train_spacy_path)

    db = get_spacy_doc(file, test_data)
    test_spacy_path = spacy_dir + '/test_data.spacy'
    db.to_disk(test_spacy_path)
    file.close()
if __name__ == '__main__':
    dataset_path = "data/jsons/Entity Recognition in Resumes.json"
    spacy_dir = 'data/spacy_format'
    dataset = trim_entity_spans(preprocess_dataturks_json(dataset_path))
    train_data, test_data = train_test_split(dataset, test_size=0.1, random_state=42)
    convert_dataturk_json_to_spacy_data_format(train_data, test_data, spacy_dir)