import re
import tensorflow as tf
import numpy as np
from datasets import load_dataset
from transformers import pipeline
from transformers import BertTokenizer, BertForMaskedLM
from transformers import TFAutoModelForMaskedLM
from transformers import AutoTokenizer

SCORE = "score"
TOKEN_STRING = "token_str"
DEFAULT_GENDER_IDENTIFIERS = [
    "she",
    "her",
    "hers",
    "woman",
    "women",
    "girl",
    "he",
    "his",
    "man",
    "men",
    "guy",
]


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

    def read_data(self):
        eval_dataset = load_dataset('myradeng/cs230-news')

    def init_LM_model(self, model_checkpoint="distilbert-base-uncased"):
        model = TFAutoModelForMaskedLM.from_pretrained(model_checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        return model, tokenizer

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
        predictions = model_fn(masked_text)
        return {prediction[TOKEN_STRING]: prediction[SCORE] for prediction in predictions}


    def mask_gender(self, tokenizer, gender_identifiers=[], input_text=""):
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
        regex = re.compile("|".join(map(re.escape, gender_identifiers)))
        return regex.sub(tokenizer.mask_token, input_text)

def main(): 
    bert = Bert()
    # TODO: make this take in many different input texts efficiently  
    masked_input = bert.mask_gender(bert.tokenizer, input_text="he can work as a lawyer")
    print(masked_input)
    print(bert.predict_mask(masked_input))

main()
