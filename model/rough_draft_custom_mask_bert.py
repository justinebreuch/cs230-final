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
        dataset = load_dataset('myradeng/cs230-news', )
        tokenized_dataset = dataset.map(lambda example: 
            self.tokenize_function(example[CONTENT_ROW]), batched = True)
        
        # Downsample if running on colab 
        downsampled_dataset = tokenized_dataset["train"].train_test_split(
            train_size = 1000, test_size  = 100, seed=42
        )

        tf_train_dataset = downsampled_dataset["train"].to_tf_dataset(
          columns=["input_ids", "attention_mask", "labels"],
          collate_fn = DataCollatorForLanguageModeling(tokenizer = self.tokenizer, mlm_probability = 0.15),
          shuffle=True,
          batch_size=32,
        )

        validation_dataset = downsampled_dataset["test"].to_tf_dataset(
          columns=["input_ids", "attention_mask", "labels"],
          collate_fn = data_collator,
          shuffle=False,
          batch_size=32,
        )

        return validation_dataset
      
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
  dataset = bert.tokenize_and_mask()
  # mask_token_index = torch.where(input_ids == bert.tokenizer.mask_token_id)[1]
  # logits = bert.model(input_ids, attention_mask=attention_mask, labels=labels).logits[0]
  # top_5_tokens = torch.topk(logits[mask_token_index, :], 5, dim=1).indices[0].tolist()
  # print(bert.tokenizer.decode(top_5_tokens))
  # bert.evaluate()
  #print(bert.predict_mask("[MASK] is a bad lawyer"))

main()
