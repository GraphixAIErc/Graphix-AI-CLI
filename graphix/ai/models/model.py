import os
from transformers import AutoTokenizer
from graphix.config.constants import MODEL_CONFIG


def load_model_and_tokenizer(model_type=None, cache_dir="./cache"):
    """
    Load a pre-trained model and tokenizer based on the specified task type.

    Args:
    - model_type (str, optional): The path to a specific model to load. If not provided, the default model for the task type will be used.

    Returns:
    - model: An instance of the loaded model class corresponding to the specified task type.
    - tokenizer: An instance of the tokenizer class associated with the loaded model.
    """

    if model_type not in MODEL_CONFIG:
        raise ValueError(f"Model type '{model_type}' is not supported.")

    model_info = MODEL_CONFIG.get(model_type, {})
    model_class = model_info.get('model', {}).get("class")
    default_model_type = model_info.get('model', {}).get("type")
    model_args = model_info.get('model', {}).get("args", {})

    model_type = model_type if model_type else default_model_type
  # Check if the model is already cached
    model_cache_path = os.path.join(cache_dir, model_type)
    if not os.path.exists(model_cache_path):
        # If the model is not cached, download it
        tokenizer = AutoTokenizer.from_pretrained(
            model_type, cache_dir=cache_dir)
        model = model_class.from_pretrained(
            model_type, cache_dir=cache_dir, **model_args)

        # Set padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id
    else:
        # Load from cache
        tokenizer = AutoTokenizer.from_pretrained(
            model_type, cache_dir=cache_dir)
        model = model_class.from_pretrained(model_type, cache_dir=cache_dir)

    return model, tokenizer
