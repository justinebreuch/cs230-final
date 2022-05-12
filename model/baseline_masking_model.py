import re
import tensorflow as tf
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, BertTokenizer, BertForMaskedLM, DataCollatorForLanguageModeling, TFAutoModelForMaskedLM, pipeline

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

class Bert():
    def __init__(self, model_checkpoint = 'bert-base-uncased'):
        """
        Instantiates model and tokenizer based on pretrained bert-base-uncased model.
        Returns:
        tokenizer -- BertTokenizer for the model
        model -- pretrained TFBertModel
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.model = BertForMaskedLM.from_pretrained(model_checkpoint)

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

    def read_eval_data(self):
        dataset = load_dataset('myradeng/cs230-news')
        # Downsample if running on colab
        downsampled_dataset = dataset["test"].train_test_split(test_size  = 100, seed=42)

        eval_dataset = downsampled_dataset["test"]
        return eval_dataset

    def compute_probs(self, eval_dataset):
        woman_probs = []
        man_probs = []

        for row in eval_dataset:
            text = row[CONTENT_ROW]
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
        clip_length = min(len(eval_df['content']), clip_length)
        probability_output = pd.DataFrame(
            {'content': eval_df['content'][0:clip_length],
             'female_probs': woman_probs[0:clip_length],
             'male_probs': man_probs[0:clip_length]
             })
        return probability_output

def main():
    bert = Bert()
    dataset = bert.read_eval_data()
    eval_dataset = bert.read_eval_data()
    print(bert.evaluate(eval_dataset))
    # masked_input = bert.mask_gender(bert.tokenizer, input_text="he can work as a lawyer")
    # print(masked_input)
    # print(bert.predict_mask(masked_input))

main()