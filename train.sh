python -m spacy train config.cfg \
    --output ./output \ 
    --paths.train spacy_data/train_data.spacy \ 
    --paths.dev spacy_data/test_data.spacy 