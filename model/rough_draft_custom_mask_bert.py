from accelerate import Accelerator
from transformers import AutoTokenizer, BertTokenizer, BertForMaskedLM, AdamW, pipeline
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import tensorflow as tf

CHUNK_SIZE = 128
SCORE = "score"
TOKEN_STRING = "token_str"
# "he" or "she" harcoded for now 
DEFAULT_GENDER_IDENTIFIERS = {2016, 2002}

class BertFinetuned():
    def __init__(self, model_checkpoint = 'bert-base-uncased'):
      self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
      self.model = BertForMaskedLM.from_pretrained(model_checkpoint) 
      
      # Allows running with mixed precision and on any kind of distributed 
      # setting 
      self.accelerator = Accelerator()
      self.optimizer = AdamW(self.model.parameters(), lr=5e-5)

    def tokenize_and_mask(self, dataset_text = []):
      input_id_tensors = []
      attention_mask_tensors = []
      labels_tensors = []

      for sentence in dataset_text:
        encoded_dict = self.tokenizer.encode_plus(
                          sentence,                     
                          add_special_tokens = True, 
                          max_length = CHUNK_SIZE,       
                          pad_to_max_length = True,
                          return_attention_mask = True,  
                          return_token_type_ids = False,
                          return_tensors = 'pt')
        
        input_ids = encoded_dict['input_ids'].tolist()[0]
        
        # Keep a copy of input ids before they are masked so that labels are 
        # preserved. Every where there is not a mask should be -100 for the 
        # cost function to work properly.
        labels = torch.tensor([-100] * len(input_ids))

        for index, input_id in enumerate(input_ids):

          # If this input id is a gender identifier, mask it but preserve its
          # label 
          if (input_id in DEFAULT_GENDER_IDENTIFIERS):
            labels[index] = input_id
            encoded_dict['input_ids'][0][index] = self.tokenizer.mask_token_id

        input_id_tensors.append(encoded_dict['input_ids'])
        attention_mask_tensors.append(encoded_dict['attention_mask'])
        labels_tensors.append(labels)

      # Flatten all features 
      features = torch.cat(input_id_tensors), torch.cat(attention_mask_tensors), torch.cat(labels_tensors)
      return features
      
    def forward(self, input_ids, attention_mask, labels):
      self.optimizer.zero_grad()
      outputs = self.model(input_ids, attention_mask = attention_mask, labels = labels)
      loss = outputs.loss

      # Calculate loss for every parameter that needs gradient update
      self.loss.backward()
      self.optimizer.step()
      
    
    def train(self, train_dataset, test_dataset, epochs, input_ids, attention_mask, labels, batch_size = 32):
       # Use GPU if we can, otherwise CPU
      device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
      self.model.to(device)

      train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
      test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=True)

      progress_bar = tqdm(train_dataloader)

      for epoch in epochs:

        # Train
        self.model.train()

        for batch in train_dataloader:
          # TODO: Use batch here instead of all input ids, attention masks and labels.
          self.forward(input_ids, attention_mask, labels)

          # Update progress bar after one batch is done
          progress_bar.update(1)

        # Evaluate
        self.model.eval()
        self.evaluate(test_dataloader)
       
    # TODO(): Finish this impl
    def evaluate(self, test_dataloader):
      losses = []
      for batch in test_dataloader:
          losses.append(self.evaluate())
      return losses 
    
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
      predict_fn = pipeline("fill-mask", model = self.model, tokenizer = self.tokenizer)
      predictions = predict_fn(masked_text)
      return {prediction[TOKEN_STRING] : prediction[SCORE] for prediction in predictions}

def main():
  bert = BertFinetuned()
  input = bert.tokenizer.mask_token + " is a lawyer";
  print(input)
  input_ids, attention_mask, labels = bert.tokenize_and_mask(["she is very cool", "he is sometimes cool too"])
  mask_token_index = torch.where(input_ids == bert.tokenizer.mask_token_id)[1]
  logits = bert.model(input_ids, attention_mask=attention_mask, labels=labels).logits[0]
  top_5_tokens = torch.topk(logits[mask_token_index, :], 5, dim=1).indices[0].tolist()
  print(bert.tokenizer.decode(top_5_tokens))
  # bert.evaluate()
  #print(bert.predict_mask("[MASK] is a bad lawyer"))

main()
