import torch
from transformers import (
    Trainer,
    get_scheduler,
    TrainingArguments,
)
from graphix.ai.data.data_loader import get_data_collator
from torch.optim import AdamW


def train(model, data, epochs, batch_size, learning_rate):
    """
    Train a DistilBERT model.
    Args:
    - model: The DistilBERT model to be trained.
    - data: The tokenized dataset.
    - epochs (int): Number of training epochs.
    - batch_size (int): Size of each training batch.
    - learning_rate (float): The learning rate for the optimizer.

    Returns:
    - model: The trained model.
    """
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for i in range(0, len(data["input_ids"]), batch_size):
            batch = {k: v[i:i+batch_size] for k, v in data.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    return model


def train_model(model_type, model, tokenizer, train_dataset, eval_dataset, epochs=3, batch_size=8, lr=5e-5, output_dir="./results"):
    data_collator = get_data_collator(model_type, tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_dir="./logs",
        logging_steps=50,
        warmup_steps=500,
        weight_decay=0.01,
        eval_strategy="epoch" if eval_dataset else "no",
        # eval_steps=50,
        # save_strategy="epoch",
        # save_steps=50,
        # load_best_model_at_end=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # optimizer
    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = len(train_dataset) // batch_size * epochs
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        optimizers=(optimizer, lr_scheduler)
    )

    # Train the model
    trainer.train()
    trainer.save_model(output_dir)
    print(f"Model fine-tuned and saved to {output_dir}")
