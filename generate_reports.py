"""This program is for generating inference report for an NER model"""
import csv
import spacy 
import pandas as pd
from data_preprocessing import train_test_split, trim_entity_spans, preprocess_dataturks_json

def separating_st_gt(test):
    """Returns the list of single texts for each documents 
    Parameters:
        test: Dataset on which report will be generated

    Returns:
        texts: List of single texts
        ground_truth_values: List of ground truth values
    """
    texts = []
    ground_truth_values = []
    for single_text in test:
        texts.append(single_text[0])
        ground_truth_values.append(list(single_text[1].values())[0])
    return texts, ground_truth_values

def entity_prediction(model_path,text):
    """Returns the prediction of each documents
    Parameters:
        text: Single texts of each documents

    Returns:
        ent.text: Predicted texts for different entity  
        ent.label: Predicted entities for different texts
    """
    nlp = spacy.load(model_path)
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def report_generation(model_pat,texts):
    """Returns the inference report.
    Parameters:
        texts: List of single texts of each documents

    Returns:
        data_for_csv(list): Combined list of all the data for the CSV file
    """
    data_for_csv = []
    pred_ent = []
    pred_label = []
    for i, text in enumerate(texts):
        entities = entity_prediction(model_path,text)
        for ent_text, ent_label in entities:
            pred_ent.append(ent_text)
            pred_label.append(ent_label)
        print("Predicted entities",entities)
        ground_truth = ground_truth_values[i]
        row = []
        for truth in ground_truth:
            start, end, entity_name = truth
            print(start, end)
            ground_truth_value = text[start:end]
            found = False
            for ent_text, ent_label in entities:
                print("Predicted Entity text & label and GT:",ent_text, ent_label, ground_truth_value)
                if ent_text == ground_truth_value and ent_label == entity_name:
                    row.append((entity_name, ent_label, ground_truth_value, ent_text, 'True'))
                    found = True
                    break
            if not found:
                row.append((entity_name, ent_label, ground_truth_value, ent_text, 'False'))
        data_for_csv.extend(row)
    return data_for_csv

if __name__ == "__main__":
    json_path='data/jsons/Entity Recognition in Resumes.json'
    csv_filename = "reports/inference_report.csv"
    model_path='models/model-best'
    data  = trim_entity_spans(preprocess_dataturks_json(json_path))
    train, test = train_test_split(data, 0.1, random_state = 100)
    texts, ground_truth_values = separating_st_gt(test)
    csv_data = report_generation(model_path,texts)
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Predicted_Entity', 'Entity_GT', 'Text_GT', 'Predicted_Text', 'Correct_Prediction'])
        writer.writerows(csv_data)
    print(f"CSV report generated: {csv_filename}")
