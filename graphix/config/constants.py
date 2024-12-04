from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)

GOOGLE_BERT = "google_bert"
GPT2 = "gpt2"

MODEL_CONFIG = {
    GOOGLE_BERT: {
        "model": {
            "class": AutoModelForSequenceClassification,
            "type": "bert-base-uncased",
            "args": {
                "num_labels": 5,
            }
        },
        "dataset": {
            "path": "yelp_review_full",
        }
    },
    GPT2: {
        "model": {
            "class": AutoModelForCausalLM,
            "type": "gpt2",
        },
        "dataset": {
            "train": "train.csv",
            "test": "test.csv",
        }
    },
}
