Named Entity Recognition on Resume Data using SpaCy
==================================================
The entities are extracted using the [SpaCy](https://spacy.io/) library. The entities are classified into the following categories:
* Name
* Email
* Phone Number
* Skills
* Degree
* College Name
* Designation
* Companies worked at
* Experience in years
* No. of projects
* Certifications
* Location

## Dataset
The dataset used for this project is the [Kaggle Resume Dataset](https://www.kaggle.com/dataturks/resume-entities-for-ner).

## Environment
The environment used for this project is as follows:

 ```bash
 pip install -r requirements.txt
 ```
 



## Approach
The approach used for this project is as follows:
* Exploratory Data Analysis (EDA) on resume data

    [EDA resume data notebook](EDA_resume.ipynb)
* The dataset is first preprocessed to remove the unnecessary columns and rows.
    ```bash
    python data_preprocessing.py
    ```
  
* The dataset is then converted into the [SpaCy](https://spacy.io/) format.
    ```bash
    python convert_dataturk_json_to_spacy_format.py
    ```
* The [SpaCy](https://spacy.io/) model is then trained on the dataset.
    ```bash
    python -m spacy train config.cfg --output models --paths.train data/spacy_format/train_data.spacy --paths.dev data/spacy_format/test_data.spacy 
    ```
* The performance of the models.
    ```bash
    python -m spacy  evaluate  models/model-best data/spacy_format/test_data.spacy --output logs/test_results.json | tee logs/evaluate_test.log

    python -m spacy  evaluate  models/model-best data/spacy_format/train_data.spacy --output logs/train_results.json | tee logs/evaluate_train.log
    ```
* The trained model is then used to extract the entities from the resume data.
    ```bash
    python inference.py
    ```

## Results
The results obtained from the trained model are as follows:
- [train dataset results](logs/evaluate_train.log)
    * NER P:   79.18 
    * NER R:   74.04 
    * NER F:   76.52 
- [test dataset results](logs/evaluate_test.log)
    * NER P:   67.61 
    * NER R:   53.02 
    * NER F:   59.43 