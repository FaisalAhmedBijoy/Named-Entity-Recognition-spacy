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

def preprocess_dataturks_json(dataturks_JSON_FilePath):
    training_data = []
    with open(dataturks_JSON_FilePath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        data = json.loads(line)
        text = data['content'].replace("\n", " ")
        entities = []
        data_annotations = data['annotation']
        if data_annotations is not None:
            for annotation in data_annotations:
                # Only a single point in text annotation.
                point = annotation['points'][0]
                labels = annotation['label']
                # Handle both list of labels or a single label.
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
                    entities.append((point_start, point_end + 1, label))
        training_data.append((text, {"entities": entities}))
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
            while valid_start < len(text) and invalid_span_tokens.match(text[valid_start]):
                valid_start += 1
            while valid_end > 1 and invalid_span_tokens.match(text[valid_end - 1]):
                valid_end -= 1
            valid_entities.append([valid_start, valid_end, label])
        cleaned_data.append([text, {'entities': valid_entities}])
    return cleaned_data



def train_test_split(data, test_size, random_state):
    random.Random(random_state).shuffle(data)
    test_idx = len(data) - math.floor(test_size * len(data))
    print(test_idx)
    train_set = data[0:test_idx]
    test_set = data[test_idx:]

    return train_set, test_set



if __name__ == "__main__":
    dataset_path = "data/jsons/Entity Recognition in Resumes.json"
    spacy_dir = 'data/spacy_format'
    dataset = trim_entity_spans(preprocess_dataturks_json(dataset_path))
    train_data, test_data = train_test_split(dataset, test_size=0.1, random_state=42)
    print("train_data len: ",len(train_data))
    print("test_data len: ",len(test_data))
