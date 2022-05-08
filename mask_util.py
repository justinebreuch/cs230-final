import re
import tensorflow as tf
from transformers import pipeline
from transformers import BertTokenizer, TFBertModel

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

pretrained_bert = "bert-base-uncased"


def setup():
    """
    Instantiates model and tokenizer based on pretrained bert-base-uncased model.
    Returns:
    tokenizer -- BertTokenizer for the model
    model -- pretrained TFBertModel
    """
    tokenizer = BertTokenizer.from_pretrained(pretrained_bert)
    model = TFBertModel.from_pretrained(pretrained_bert)
    return tokenizer, model


def predict_mask(model, tokenizer, masked_text):
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
    model = pipeline("fill-mask", model=pretrained_bert)
    predictions = model(masked_text)
    return {prediction[TOKEN_STRING]: prediction[SCORE] for prediction in predictions}


def mask_gender(tokenizer, gender_identifiers=[], input_text=""):
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


tokenizer, model = setup()
masked_input = mask_gender(tokenizer, input_text="he can work as a lawyer")
print(masked_input)
print(predict_mask(model, tokenizer, masked_input))
