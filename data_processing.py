import json
import re
import json
from pprint import pprint
import re 
import math
import json
import spacy
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from spacy.tokens import DocBin
from spacy.util import filter_spans
def preprocess_dataturks_json(dataturks_JSON_FilePath):
    training_data = []
    lines=[]
    with open(dataturks_JSON_FilePath, 'r',encoding='utf-8') as f:
        lines = f.readlines()
        # print(lines)

    for line in lines:
        data = json.loads(line)
        text = data['content'].replace("\n", " ")
        entities = []
        data_annotations = data['annotation']
        if data_annotations is not None:
            for annotation in data_annotations:
                #only a single point in text annotation.
                point = annotation['points'][0]
                labels = annotation['label']
                # handle both list of labels or a single label.
                if not isinstance(labels, list):
                    labels = [labels]

                for label in labels:
                    point_start = point['start']
                    point_end = point['end']
                    point_text = point['text']

                    lstrip_diff = len(point_text) - len(point_text.lstrip())
                    rstrip_diff = len(point_text) - len(point_text.rstrip())
                    if lstrip_diff != 0:
                        point_start = point_start + lstrip_diff
                    if rstrip_diff != 0:
                        point_end = point_end - rstrip_diff
                    entities.append((point_start, point_end + 1 , label))
        training_data.append((text, {"entities" : entities}))
    return training_data

def trim_entity_spans(data: list) -> list:
    """Removes leading and trailing white spaces from entity spans.

    Args:
        data (list): The data to be cleaned in spaCy JSON format.

    Returns:
        list: The cleaned data.
    """
    invalid_span_tokens = re.compile(r'\s')

    cleaned_data = []
    for text, annotations in data:
        entities = annotations['entities']
        valid_entities = []
        for start, end, label in entities:
            valid_start = start
            valid_end = end
            while valid_start < len(text) and invalid_span_tokens.match(
                    text[valid_start]):
                valid_start += 1
            while valid_end > 1 and invalid_span_tokens.match(
                    text[valid_end - 1]):
                valid_end -= 1
            valid_entities.append([valid_start, valid_end, label])
        cleaned_data.append([text, {'entities': valid_entities}])
    return cleaned_data
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
                span = doc.char_span(start, end, label = label, alignment_mode = "strict")
            except:
                continue
            
            if span is None:
                err_data = str([start, end])+ " " + str(text) + "\n"
                file.write(err_data)
                
            else:
                ents.append(span)
        try:
            doc.ents = ents
            db.add(doc)
        except:
            pass
    return db
def train_test_split(data, test_size, random_state):

    random.Random(random_state).shuffle(data)
    test_idx = len(data) - math.floor(test_size * len(data))
    train_set = data[0: test_idx]
    test_set = data[test_idx: ]

    return train_set, test_set
def convert_dataturk_json_to_spacy_data_format(train_data, test_data,spacy_dir):
    file = open("sample.txt", "w")
    db = get_spacy_doc(file, train_data)
    train_spacy_path=spacy_dir+'/train_data.spacy'
    db.to_disk(train_spacy_path)

    db = get_spacy_doc(file, test_data)
    test_spacy_path=spacy_dir+'/test_data.spacy'
    db.to_disk(test_spacy_path)
    file.close()
if __name__=="__main__":
    dataset_path = "data/jsons/Entity Recognition in Resumes.json"
    spacy_dir='data/spacy_format'
    # dataset = preprocess_dataturks_json(dataset_path)
    # dataset = trim_entity_spans(dataset)
    dataset=trim_entity_spans(preprocess_dataturks_json(dataset_path))
    train_data, test_data = train_test_split(dataset, test_size = 0.1, random_state = 42)
    convert_dataturk_json_to_spacy_data_format(train_data, test_data,spacy_dir)
   
   