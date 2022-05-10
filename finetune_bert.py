from transformers import TFAutoModelForMaskedLM
from transformers import AutoTokenizer
import numpy as np
import tensorflow as tf
from transformers import DataCollatorForLanguageModeling
from huggingface_hub import notebook_login
from transformers import create_optimizer
from transformers.keras_callbacks import PushToHubCallback
import tensorflow as tf
from datasets import load_dataset

CHUNK_SIZE = 128

def init_model(model_checkpoint="distilbert-base-uncased"):
    model = TFAutoModelForMaskedLM.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    return model, tokenizer

def load_test_imdb_dataset():
	imdb_dataset = load_dataset("imdb")
	return imdb_dataset

def tokenize_function(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

def group_texts(examples):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // CHUNK_SIZE) * CHUNK_SIZE
    # Split by chunks of max_len
    result = {
        k: [t[i : i + CHUNK_SIZE] for i in range(0, total_length, CHUNK_SIZE)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result

def main():
	model,tokenizer = init_model()
	imdb_dataset = load_test_imdb_dataset()
	# Use batched=True to activate fast multithreading!
	tokenized_datasets = imdb_dataset.map(
	    tokenize_function, batched=True, remove_columns=["text", "label"]
	)
	lm_datasets = tokenized_datasets.map(group_texts, batched=True)
	data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

	train_size = 10_000
	test_size = int(0.1 * train_size)

	# Downsample if running on colab 
	downsampled_dataset = lm_datasets["train"].train_test_split(
    	train_size=train_size, test_size=test_size, seed=42
	)

	# Hugging face login
	notebook_login()

	tf_train_dataset = downsampled_dataset["train"].to_tf_dataset(
    columns=["input_ids", "attention_mask", "labels"],
    collate_fn=data_collator,
    shuffle=True,
    batch_size=32,
	)

	tf_eval_dataset = downsampled_dataset["test"].to_tf_dataset(
	    columns=["input_ids", "attention_mask", "labels"],
	    collate_fn=data_collator,
	    shuffle=False,
	    batch_size=32,
	)

	num_train_steps = len(tf_train_dataset)
	optimizer, schedule = create_optimizer(
	    init_lr=2e-5,
	    num_warmup_steps=1_000,
	    num_train_steps=num_train_steps,
	    weight_decay_rate=0.01,
	)
	model.compile(optimizer=optimizer)

	model.fit(tf_train_dataset, validation_data=tf_eval_dataset)
	eval_loss = model.evaluate(tf_eval_dataset)