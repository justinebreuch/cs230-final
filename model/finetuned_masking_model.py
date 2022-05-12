import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForMaskedLM, DataCollatorForLanguageModeling, BertForMaskedLM, TFPreTrainedModel, create_optimizer
from huggingface_hub import notebook_login
from transformers.keras_callbacks import PushToHubCallback
from datasets import load_dataset
import pandas as pd

CHUNK_SIZE = 128
DEFAULT_TRAIN_SIZE = 1000

CONTENT_ROW = "content"
SCORE = "score"
TOKEN_STRING = "token_str"

DEFAULT_GENDER_IDENTIFIERS = [
		"she",
		"her",
		"hers",
		"woman",
		"women",
		"female",
		"he",
		"his",
		"him",
		"man",
		"men",
		"male",
]

WOMAN_KEYWORDS = ['woman', 'women', 'female', 'she', 'her', 'hers']
MAN_KEYWORDS = ['man', 'men', 'male', 'he', 'his', 'him']

TOP_K = 100

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

		self.model.compile(optimizer = optimizer)
		self.model.fit(train_dataset, validation_data = eval_dataset)

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

	def predict_mask(self, masked_text):
		"""
    Provide predictions and scores for masked tokens.
    Arguments:
      model -- model to use for prediction
      tokenizer -- tokenizer to identify the token_str
      masked_text -- input text to run predictions on
    Returns:
      predictions -- dictionary of { predictions : probability }
    Example: ("[Mask] should be president!") : {'she' : 0.50, 'he': 0.5}
    """
		model_fn = pipeline("fill-mask", model = self.model, tokenizer = self.tokenizer)
		predictions = model_fn(masked_text, top_k=TOP_K)
		return predictions

	def mask_gender(self, gender_identifiers=[], input_text=""):
		"""
    Masks the input text with the mask_token for the given tokenizer
    Arguments:
      tokenizer -- tokenizer to identify the token_str
      gender_identifiers (optional) -- list of identifiers to mask (i.e. ["Megan", "boy", "guy"])
      input_text -- the string to mask
    Returns:
      output_text -- masked version of the input_text
    Example: ("[Mask] should be president!") : {'she' : 0.50, 'he': 0.5}
    """
		if not gender_identifiers:
			gender_identifiers = DEFAULT_GENDER_IDENTIFIERS
		regex = re.compile(r'\b(?:%s)\b' % '|'.join(gender_identifiers))
		return regex.sub(self.tokenizer.mask_token, input_text)

	def split_to_contexts(self, eval_dataset, context_size = 100):
		concat_text = ' '.join(eval_dataset)
		words = concat_text.split()
		grouped_words = [' '.join(words[i: i + context_size]) for i in range(0, len(words), context_size)]
		print(grouped_words)
		return grouped_words

	def read_eval_data(self):
		dataset = load_dataset('myradeng/cs230-news')

		# Downsample if running on colab
		downsampled_dataset = dataset["test"].train_test_split(test_size  = 100, seed=42)
		eval_dataset = downsampled_dataset["test"]
		print(eval_dataset)
		repartitioned = self.split_to_contexts(eval_dataset[CONTENT_ROW])
		print(repartitioned)
		return repartitioned

	def compute_probs(self, eval_dataset):
		woman_probs = []
		man_probs = []

		for row in eval_dataset:
			text = row
			print(text)
			masked_input = self.mask_gender(input_text = text)
			print(masked_input)
			predictions = self.predict_mask(masked_input)
			print(predictions)
			if len(predictions) != TOP_K:
				for predictions_list in predictions:
					woman_prob_numerator = 0
					man_prob_numerator = 0
					all_gender_denominator = 0
					for prediction in predictions_list:
						if prediction[TOKEN_STRING] in WOMAN_KEYWORDS:
							woman_prob_numerator += prediction[SCORE]
							all_gender_denominator += prediction[SCORE]
						if prediction[TOKEN_STRING] in MAN_KEYWORDS:
							man_prob_numerator += prediction[SCORE]
							all_gender_denominator += prediction[SCORE]
					if all_gender_denominator == 0:
						woman_probs.append(0)
						man_probs.append(0)
					else:
						woman_probs.append(woman_prob_numerator / all_gender_denominator)
						man_probs.append(man_prob_numerator / all_gender_denominator)
						assert((woman_prob_numerator / all_gender_denominator) +
									 (man_prob_numerator / all_gender_denominator) == 1.0)
			else:
				woman_prob_numerator = 0
				man_prob_numerator = 0
				all_gender_denominator = 0
				for prediction in predictions:
					if prediction[TOKEN_STRING] in WOMAN_KEYWORDS:
						woman_prob_numerator += prediction[SCORE]
						all_gender_denominator += prediction[SCORE]
					if prediction[TOKEN_STRING] in MAN_KEYWORDS:
						man_prob_numerator += prediction[SCORE]
						all_gender_denominator += prediction[SCORE]
				if all_gender_denominator == 0:
					woman_probs.append(0)
					man_probs.append(0)
				else:
					woman_probs.append(woman_prob_numerator / all_gender_denominator)
					man_probs.append(man_prob_numerator / all_gender_denominator)
					assert((woman_prob_numerator / all_gender_denominator) +
								 (man_prob_numerator / all_gender_denominator) == 1.0)
		print("Woman probs: " + str(woman_probs))
		print("Man probs: " + str(man_probs))
		return woman_probs, man_probs

	# Example model usage from a data file path, returns dataframe with probabilities
	def evaluate(self, eval_df):
		woman_probs, man_probs = self.compute_probs(eval_df)

		# For some reason woman_probs, man_probs, and content are different
		# different lengths.
		clip_length = min(len(woman_probs), len(man_probs))
		clip_length = min(len(eval_df), clip_length)
		probability_output = pd.DataFrame(
				{'content': eval_df[0:clip_length],
				 'female_probs': woman_probs[0:clip_length],
				 'male_probs': man_probs[0:clip_length]
				 })
		return probability_output

def main():
	# Hugging face login
	notebook_login()
	bert = BertFinetuned()
	train_dataset, eval_dataset = bert.load_dataset(train_size = 100, test_size = 10)
	bert.fit(train_dataset, eval_dataset)
	loss = bert.evaluate(eval_dataset)
	print(loss)
main()