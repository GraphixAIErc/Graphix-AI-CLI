import requests
import pandas as pd
from io import StringIO
from datasets import load_dataset, Dataset
from graphix.config.settings import SERVER_URL

from transformers import (
    DistilBertTokenizer,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling
)
from graphix.config.constants import (
    MODEL_CONFIG,
    GOOGLE_BERT,
    GPT2
)


def load_data(file_path):
    """
    Load data from a CSV file.
    Args:
    - file_path (str): The file path to the CSV file.

    Returns:
    - data (pd.DataFrame): The loaded data.
    """
    data = pd.read_csv(file_path)
    return data


def load_dataset_from_csv(file_path):
    df = pd.read_csv(file_path)
    return Dataset.from_pandas(df)


def prepare_data(data, tokenizer_path):
    """
    Tokenize and prepare data using DistilBERT tokenizer.
    Args:
    - data (pd.DataFrame): The data to be tokenized.
    - tokenizer_path (str): Path to the tokenizer.

    Returns:
    - tokenized_data (dict): Tokenized data prepared for DistilBERT input.
    """
    tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
    tokenized_data = tokenizer(data["text"].tolist(
    ), add_special_tokens=True, truncation=True, padding=True, return_tensors="pt")
    return tokenized_data


def get_train_and_eval_splits(tokenized_dataset):
    """
    Extract and prepare training and evaluation splits from a tokenized dataset.

    Args:
    - tokenized_dataset (dict): A dictionary containing tokenized data splits (e.g., "train", "test", "validation", etc.).

    Returns:
    - train_dataset (dict): The tokenized training dataset prepared for input to the model.
    - eval_dataset (dict): The tokenized evaluation dataset prepared for input to the model.
      This will be the first available split found among "test", "validation", "eval", or "dev".
    """
    train_dataset = None
    eval_dataset = None

    if "train" not in tokenized_dataset:
        train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
        train_dataset = train_test_split["train"]
        eval_dataset = train_test_split["test"]
    else:
        available_splits = tokenized_dataset.keys()
        train_dataset = tokenized_dataset.get("train")

        for split_name in ["test", "validation", "eval", "dev"]:
            if split_name in available_splits:
                eval_dataset = tokenized_dataset[split_name]
                break

        if train_dataset is None:
            raise ValueError("No training dataset found.")

    return train_dataset, eval_dataset


def get_data_collator(model_type, tokenizer):
    """
    Get the appropriate data collator for the specified task type.

    Args:
    - model_type (str): The model type of task for which the data collator is needed.
                       It determines which collator to use based on the model being fine-tuned (e.g., BERT or GPT-2).
    - tokenizer: The tokenizer associated with the model, which is used for padding and formatting the input data.

    Returns:
    - Data collator: An instance of a data collator that prepares batches of data for training.
                     The collator will handle padding and other necessary transformations based on the task type.
    """

    def google_bert():
        return DataCollatorWithPadding(tokenizer=tokenizer)

    def gpt2():
        return DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

    model_types = {
        GOOGLE_BERT: google_bert,
        GPT2: gpt2,
    }

    collator_function = model_types.get(model_type)

    if collator_function:
        return collator_function()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def load_and_preprocess_dataset(model_type, tokenizer, training_dataset_url, validation_dataset_url):
    """
    Load and preprocess a dataset for a specific task type using the DistilBERT tokenizer.

    Args:
    - model_type (str): The model type of task for which the dataset is prepared (e.g., "text", "generation", etc.).
    - tokenizer (transformers.PreTrainedTokenizer): The tokenizer to be used for processing the dataset.
    - training_dataset_url (str): The file path to the dataset. If not provided, a default path from the MODEL_CONFIG is used.
    - validation_dataset_url (str): The file path to the dataset. If not provided, a default path from the MODEL_CONFIG is used.

    Returns:
    - tokenized_dataset (Dataset): A tokenized dataset prepared for DistilBERT input. The dataset is processed based on the specified task type.
    """
    model_info = MODEL_CONFIG.get(model_type, {})
    dataset_type = model_info.get("dataset", {}).get("type")

    def google_bert():
        default_dataset_path = model_info.get("dataset", {}).get("path")
        dataset_args = model_info.get("dataset", {}).get("args", {})

        dataset_path = dataset_path if dataset_path else default_dataset_path

        dataset = load_dataset(dataset_path, **dataset_args)
        tokenized_dataset = dataset.map(lambda examples: tokenizer(
            examples["text"], padding="max_length", truncation=True), batched=True)
        return get_train_and_eval_splits(tokenized_dataset)

    def gpt2():
        train_response = requests.get(training_dataset_url)
        test_dataset = requests.get(validation_dataset_url)
        train_df = pd.read_csv(StringIO(train_response.text))
        test_df = pd.read_csv(StringIO(test_dataset.text))
        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)
        tokenized_train_dataset = train_dataset.map(lambda examples: tokenizer(
            examples['text'], padding="max_length", max_length=512, return_tensors=None, truncation=True), batched=True)
        tokenized_test_dataset = test_dataset.map(lambda examples: tokenizer(
            examples['text'], padding="max_length", max_length=512, return_tensors=None, truncation=True), batched=True)

        return tokenized_train_dataset, tokenized_test_dataset

    model_types = {
        GOOGLE_BERT: google_bert,
        GPT2: gpt2,
    }

    dataset_function = model_types.get(model_type)

    if dataset_function:
        return dataset_function()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")