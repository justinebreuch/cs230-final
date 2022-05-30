# Studying Gender Bias in News Using BERT Masked Language Models (CS230, Spring 2022)

**Models**: Main model code is in *model/*, including baseline and finetuned BERT models. Each model class supports initializing the model, masking, tokenization, and functions required for cosine embedding analysis and masked language model prediction.

**Sample data**: Some sample data is included in *data/*, though actual data used is on Hugging Face [https://huggingface.co/datasets/myradeng/cs-230-news-v3]

**Data Pre-processing**: Relevant functions are included in build_data.py and preprocess_data.py for cleaning the input data and splitting into train/dev/test. Examples for how we processed data are in process_news_data_final.ipynb

**Cosine Similarity Word Embedding Analysis**: Reference workflow is documented in *cosine_word_embeddings_analysis.ipynb*

**Masked Language Gender Word Prediction**: Reference workflow for inference is documented in *eval_broken_down_by_political_learning.ipynb*
