import tensorflow as tf
import pandas as pd
import os
import shutil
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures

DATA_COLUMN = 'DATA_COLUMN';
LABEL_COLUMN = 'LABEL_COLUMN';

def load_dataset():
  url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

  return tf.keras.utils.get_file(fname="aclImdb_v1.tar.gz", 
                                    origin=url,
                                    untar=True,
                                    cache_dir='.',
                                    cache_subdir='')
def create_directory():
  dataset = load_dataset();
  main_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
  train_dir = os.path.join(main_dir, 'train')
  remove_dir = os.path.join(train_dir, 'unsup')
  shutil.rmtree(remove_dir)

# Split between train and test with an 80/20 split.
def split_dataset():
  train = tf.keras.preprocessing.text_dataset_from_directory(
      'aclImdb/train', batch_size=30000, validation_split=0.2, 
      subset='training', seed=123)
  test = tf.keras.preprocessing.text_dataset_from_directory(
      'aclImdb/test', batch_size=30000, validation_split=0.2, 
      subset='validation', seed=123)
  return train, test

def load_features_and_labels(dataset):
  for i in dataset.take(1):
    features = i[0].numpy()
    labels = i[1].numpy()

    dataset = pd.DataFrame([features, labels]).T
    dataset.columns = [DATA_COLUMN, LABEL_COLUMN]
    dataset[DATA_COLUMN] = dataset[DATA_COLUMN].str.decode("utf-8")
    return dataset.head()

def convert_data_to_examples(train, test): 
  train_examples = train.apply(lambda x: InputExample(guid=None, 
                               text_a = x[DATA_COLUMN], 
                               text_b = None,
                               label = x[LABEL_COLUMN]), axis = 1)

  validation_examples = test.apply(lambda x: InputExample(guid=None, 
                                   text_a = x[DATA_COLUMN], 
                                   text_b = None,
                                   label = x[LABEL_COLUMN]), axis = 1)
  return train_examples, validation_examples

def convert_examples_to_tf_dataset(examples, tokenizer, max_length=128):
    features = [] 

    for e in examples:
        input_dict = tokenizer.encode_plus(
            e.text_a,
            add_special_tokens=True,
            max_length=max_length, # truncates if len(s) > max_length
            return_token_type_ids=True,
            return_attention_mask=True,
            padding='longest', 
            truncation=True,
        )

        input_ids, token_type_ids, attention_mask = (input_dict["input_ids"],
            input_dict["token_type_ids"], input_dict['attention_mask'])
        
        print("preprocessing as tensors")
        print(input_dict)

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=e.label
            )
        )

    def gen():
        for feature in features:
            yield (
                {
                    "input_ids": feature.input_ids,
                    "attention_mask": feature.attention_mask,
                    "token_type_ids": feature.token_type_ids,
                },
                feature.label,
            )
    # I tried using from_tensor_slices so many times so avoid all this unnecessary code and it 
    # wasn't working.
    return tf.data.Dataset.from_generator(
        gen,
        ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
                "token_type_ids": tf.TensorShape([None]),
            },
            tf.TensorShape([]),
        ),
    )

def setup():
  load_dataset();
  create_directory();
  train, test = split_dataset();
  train = load_features_and_labels(train);
  test = load_features_and_labels(test);
  return train, test 

def convert_data():
  train, test = setup();
  return convert_data_to_examples(train, test)

def train_model():
  model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
  tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
  train_examples, validation_examples = convert_data();
  train_data = convert_examples_to_tf_dataset(list(train_examples), tokenizer)
  train_data = train_data.shuffle(100).batch(32).repeat(2)

  validation_data = convert_examples_to_tf_dataset(list(validation_examples), tokenizer)
  validation_data = validation_data.batch(32)
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])

  model.fit(train_data, epochs=2, validation_data=validation_data)
  return model, tokenizer

def predict():
  model, tokenizer = train_model();

  pred_sentences = [
                    'This was a great movie',
                    'One of the worst movies I have ever seen. Bad bad bad'] 

  tf_batch = tokenizer(pred_sentences, max_length=128, padding='longest', truncation=True, return_tensors='tf')
  tf_outputs = model(tf_batch)
  tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)
  print("really bad predictions are: ")
  print(tf_predictions)
  labels = ['Negative','Positive']
  label = tf.argmax(tf_predictions, axis=1)
  label = label.numpy()
  for i in range(len(pred_sentences)):
    print(pred_sentences[i], ": \n", labels[label[i]])

predict();
