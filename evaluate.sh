python -m spacy  evaluate  models/model-best data/spacy_format/test_data.spacy --output logs/test_results.json | tee logs/evaluate_test.log
python -m spacy  evaluate  models/model-best data/spacy_format/train_data.spacy --output logs/train_results.json | tee logs/evaluate_train.log