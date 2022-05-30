import numpy as np
import tensorflow as tf
from transformers import (
    AutoTokenizer,
    TFAutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    BertForMaskedLM,
    TFPreTrainedModel,
    create_optimizer,
    pipeline,
)
from huggingface_hub import notebook_login
from transformers.keras_callbacks import PushToHubCallback
from datasets import load_dataset
import pandas as pd
import re

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

WOMAN_KEYWORDS = ["woman", "women", "female", "she", "her", "hers"]
MAN_KEYWORDS = ["man", "men", "male", "he", "his", "him"]

TOP_K = 100


class BertFinetuned:
    def __init__(self, model_checkpoint="bert-base-uncased"):
        """
        Instantiates model and tokenizer based on pretrained bert-base-uncased model.
        Returns:
        tokenizer -- AutoTokenizer for the model
        model -- pretrained BertForMaskedLM
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.model = TFAutoModelForMaskedLM.from_pretrained(model_checkpoint)
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm_probability=0.15
        )

    def load_dataset(self, train_size=None, test_size=None, batch_size=32):
        dataset = load_dataset("myradeng/cs230-news")

        train_size = train_size if train_size is not None else len(dataset)
        test_size = test_size if test_size is not None else int(0.1 * train_size)

        tokenized_datasets = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=[
                "Unnamed: 0",
                "id",
                "title",
                "publication",
                "author",
                "date",
                "year",
                "month",
                "url",
                "label",
                "content",
                "content_list",
            ],
        )

        # Re-partition dataset into CHUNK-SIZES
        tokenized_datasets = tokenized_datasets.map(self.group_texts, batched=True)

        # Downsample if running on colab
        downsampled_dataset = tokenized_datasets["train"].train_test_split(
            train_size=train_size, test_size=test_size, seed=42
        )

        train_dataset = downsampled_dataset["train"].to_tf_dataset(
            columns=["input_ids", "attention_mask", "labels"],
            collate_fn=self.data_collator,
            shuffle=True,
            batch_size=batch_size,
        )

        eval_dataset = downsampled_dataset["test"].to_tf_dataset(
            columns=["input_ids", "attention_mask", "labels"],
            collate_fn=self.data_collator,
            shuffle=False,
            batch_size=batch_size,
        )

        return train_dataset, eval_dataset

    def tokenize_function(self, examples):
        result = self.tokenizer(
            examples[CONTENT_ROW], padding="max_length", max_length=512
        )
        result["labels"] = result["input_ids"].copy()
        if self.tokenizer.is_fast:
            result["word_ids"] = [
                result.word_ids(i) for i in range(len(result["input_ids"]))
            ]
        return result

    def fit(self, train_dataset, eval_dataset, batch_size=32):
        optimizer, _ = create_optimizer(
            init_lr=2e-5,
            num_warmup_steps=1_000,
            num_train_steps=len(train_dataset),
            weight_decay_rate=0.01,
        )

        self.model.compile(optimizer=optimizer)
        self.model.fit(
            train_dataset,
            validation_data=eval_dataset,
            batch_size=batch_size,
            verbose=1,
        )

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
        model_fn = pipeline("fill-mask", model=self.model, tokenizer=self.tokenizer)
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
        regex = re.compile(r"\b(?:%s)\b" % "|".join(gender_identifiers))
        return regex.sub(self.tokenizer.mask_token, input_text)

    def mask_single_gender(self, gender_identifiers=[], input_text=""):
        """
        Masks the input text with the mask_token for the given tokenizer.
        Chooses single, center-most relevant token to mask.
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
        regex = re.compile(r"\b(?:%s)\b" % "|".join(gender_identifiers))
        matches = list(re.finditer(regex, input_text.lower()))

        middle_index = len(input_text) / 2
        single_match_start = 0
        single_match_end = 0
        min_distance = 10000

        if len(matches) == 0:
            return input_text
        elif len(matches) == 1:
            single_match_start = matches[0].start()
            single_match_end = matches[0].end()
        else:
            match_indices = []
            for match in matches:
                match_indices.append((match.start(), match.end()))
            for match_index_tuple in match_indices:
                match_index = int((match_index_tuple[0] + match_index_tuple[1]) / 2)
                current_distance = abs(match_index - middle_index)
                if current_distance < min_distance:
                    min_distance = current_distance
                    single_match_start = match_index_tuple[0]
                    single_match_end = match_index_tuple[1]

        label = input_text[single_match_start:single_match_end].strip()
        input_text = (
            input_text[:single_match_start] + "[MASK]" + input_text[single_match_end:]
        )
        return input_text, label

    def split_to_contexts(self, eval_dataset, context_size=100):
        concat_text = " ".join(eval_dataset)
        words = concat_text.split()
        grouped_words = [
            " ".join(words[i : i + context_size])
            for i in range(0, len(words), context_size)
        ]
        print(grouped_words)
        return grouped_words

    def read_eval_data(self, dataset, downsample=False):
        eval_dataset = dataset["validation"]
        # Downsample if running on colab
        if downsample:
            downsampled_dataset = dataset["validation"].train_test_split(
                test_size=100, seed=42
            )
            eval_dataset = downsampled_dataset["test"]
        repartitioned = self.split_to_contexts(eval_dataset[CONTENT_ROW])
        eval_dataset_df = pd.DataFrame({"content": repartitioned})
        return eval_dataset_df

    def compute_single_prob(self, predictions):
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
            woman_prob = 0
            man_prob = 0
        else:
            woman_prob = woman_prob_numerator / all_gender_denominator
            man_prob = man_prob_numerator / all_gender_denominator
            assert woman_prob + man_prob == 1.0
        return woman_prob, man_prob

    def compute_probs(self, predictions):
        """
        Computes normalized gender probability given a list of predictions
        (corresponding to a single context)
        Arguments:
          predictions -- list of predictions output for a single context
        Returns:
          output_text -- woman_prob, man_prob
        """
        woman_prob = 0
        man_prob = 0
        if len(predictions) != TOP_K:
            woman_prob_list = []
            man_prob_list = []
            for prediction in predictions:
                woman_prob, man_prob = self.compute_single_prob(prediction)
                woman_prob_list.append(woman_prob)
                man_prob_list.append(man_prob)
            woman_prob = np.mean(woman_prob_list)
            man_prob = np.mean(man_prob_list)
        else:
            woman_prob, man_prob = self.compute_single_prob(predictions)
        return woman_prob, man_prob

    def evaluate(self, eval_df):
        model_fn = pipeline("fill-mask", model=self.model, tokenizer=self.tokenizer)
        predictions = []
        woman_probs = []
        man_probs = []
        for prediction in tqdm(model_fn(KeyDataset(eval_df, "content"), top_k=TOP_K)):
            (woman_prob, man_prob) = bert.compute_probs(prediction)
            woman_probs.append(woman_prob)
            man_probs.append(man_prob)

        probability_output = pd.DataFrame(
            {
                "content": eval_df["content"],
                "label": eval_df["label"],
                "female_probs": woman_probs,
                "male_probs": man_probs,
            }
        )

        return probability_output

    def get_context_indices(self, context, gender_keywords, target_keywords):
        # masked_context = self.mask_single_gender(gender_keywords, context)
        masked_context = context.split()
        # if "[MASK]" not in masked_context[0]:
        if "[MASK]" not in masked_context:
            return None

        gender_indices = []
        target_indices = []

        # masked_context = masked_context[0].split()
        for i in range(0, len(masked_context)):
            if masked_context[i].lower().strip() in target_keywords:
                target_indices.append(i)
            if masked_context[i] == "[MASK]":
                gender_indices.append(i)
        return gender_indices, target_indices

    def get_cosine_similarities(self, context, label, gender_keywords, target_keywords):
        """
        Computes cosine similarities between gender_keyword and target_keywords
        Arguments:
          context -- input context (non-masked)
          gender_keywords -- list of gender keywords e.g. woman keywords
          target_keywords -- list of target keywords e.g. strength keywords
        Returns:
          mean cosine similarity or None if no keyword match
        """
        tok = self.tokenizer(context, return_tensors="pt")
        sent_idxs = self.get_context_indices(context, gender_keywords, target_keywords)
        if sent_idxs is None:
            return None
        gender_index = sent_idxs[0][0]
        target_indices = sent_idxs[1]

        cosine_sims = dict()
        context_list = context.split()
        for target_index in target_indices:
            tok_ids = [
                np.where(np.array(tok.word_ids()) == idx)
                for idx in [gender_index, target_index]
            ]
            target_word = context_list[target_index]
            gender_word = label

            with torch.no_grad():
                out = self.model(**tok)

            # Only grab the last hidden state
            states = out.hidden_states[-1].squeeze()
            embs = states[[tup[0][0] for tup in tok_ids]]

            pronoun_embedding = embs[0].reshape(1, -1)
            target_embedding = embs[1].reshape(1, -1)

            cosine_sim = torch.cosine_similarity(pronoun_embedding, target_embedding)
            if (gender_word, target_word) not in cosine_sims:
                cosine_sims[(gender_word.lower(), target_word.lower())] = []
                cosine_sims[(gender_word.lower(), target_word.lower())].append(
                    cosine_sim.item()
                )
            else:
                cosine_sims[(gender_word.lower(), target_word.lower())].append(
                    cosine_sim.item()
                )
        if len(cosine_sims) == 0:
            return None
        return cosine_sims


def main():
    # Hugging face login
    notebook_login()
    bert = BertFinetuned()
    # Modify this
    train_dataset, eval_dataset = bert.load_dataset(train_size=10, test_size=1)
    bert.fit(train_dataset, eval_dataset)
    loss = bert.evaluate(bert.read_eval_data())
    print(loss)

    # Inference
    dataset = bert.read_eval_data(dataset, True)
    for idx, row in dataset.iterrows():
        output = mask_single_gender(input_text=row["content"])
        dataset.loc[idx, "content"] = output[0]
        dataset.loc[idx, "label"] = output[1]
    dataset = Dataset.from_pandas(dataset)
    probability_output_df = bert.evaluate(dataset)

    # Example for computing cosine similarities
    # bert = Bert()

    # data_files = {"train": "train.csv", "test": "dev.csv", "validation": "test.csv"}
    # dataset = load_dataset("myradeng/cs-230-news-v3", data_files=data_files)

    # dataset = bert.read_eval_data(dataset, True)
    # cosine_sims = []
    # for idx, row in dataset.iterrows():
    #   cosine_sim = bert.get_cosine_similarities(row["content"], MAN_KEYWORDS, STRENGTH)
    #   if cosine_sim is not None:
    #     cosine_sims.append(cosine_sim)

    # print(cosine_sims)


main()
