import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForMaskedLM, DataCollatorForLanguageModeling, BertForMaskedLM, TFPreTrainedModel, create_optimizer
from huggingface_hub import notebook_login
from transformers.keras_callbacks import PushToHubCallback
from datasets import load_dataset

CHUNK_SIZE = 128
CONTENT_ROW = "content"
DEFAULT_TRAIN_SIZE = 1000

class BertFinetuned():
	def __init__(self, model_checkpoint = 'bert-base-uncased'):
		"""
    Instantiates model and tokenizer based on pretrained bert-base-uncased model.
    Returns:
    tokenizer -- AutoTokenizer for the model
    model -- pretrained BertForMaskedLM
    """
		self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
		self.model = TFAutoModelForMaskedLM.from_pretrained(model_checkpoint)
		self.data_collator = DataCollatorForLanguageModeling(tokenizer = self.tokenizer, mlm_probability=0.15)

	def load_dataset(self, train_size = None , test_size = None, batch_size = 32):
		dataset = load_dataset('myradeng/cs230-news')

		train_size = train_size if train_size is not None else len(dataset)
		test_size =  test_size if test_size is not None else int(0.1 * train_size)

		tokenized_datasets = dataset.map(
				self.tokenize_function,
				batched = True,
				remove_columns=["Unnamed: 0", "id",	"title", "publication", "author", "date", "year", "month", "url", "label",  "content", "content_list"]
		)

		# Re-partition dataset into CHUNK-SIZES
		tokenized_datasets = tokenized_datasets.map(self.group_texts, batched = True)

		# Downsample if running on colab
		downsampled_dataset = tokenized_datasets["train"].train_test_split(
				train_size=train_size, test_size=test_size, seed=42
		)

		train_dataset = downsampled_dataset["train"].to_tf_dataset(
				columns=["input_ids", "attention_mask", "labels"],
				collate_fn = self.data_collator,
				shuffle = True,
				batch_size = batch_size,
		)

		eval_dataset = downsampled_dataset["test"].to_tf_dataset(
				columns=["input_ids", "attention_mask", "labels"],
				collate_fn = self.data_collator,
				shuffle = False,
				batch_size = batch_size,
		)

		return train_dataset, eval_dataset

	def tokenize_function(self, examples):
		result = self.tokenizer(examples[CONTENT_ROW], padding = 'max_length', max_length = 512)
		result["labels"] = result["input_ids"].copy()
		if self.tokenizer.is_fast:
			result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
		return result

	def fit(self, train_dataset, eval_dataset):
		optimizer, _ = create_optimizer(
				init_lr=2e-5,
				num_warmup_steps=1_000,
				num_train_steps = len(train_dataset),
				weight_decay_rate=0.01,
		)
		self.model
		self.model.compile(optimizer = optimizer)
		self.model.fit(train_dataset, validation_data = eval_dataset)

	def evaluate(self, eval_dataset):
		return self.model.evaluate(eval_dataset)

	def group_texts(self, examples):
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
	# Hugging face login
	notebook_login()
	bert = BertFinetuned()
	train_dataset, eval_dataset = bert.load_dataset(train_size = 100, test_size = 10)
	bert.fit(train_dataset, eval_dataset)
	loss = bert.evaluate(eval_dataset)

main()