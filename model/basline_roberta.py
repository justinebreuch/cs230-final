import re
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from re import search
from tqdm.auto import tqdm
import torch
from transformers.pipelines.pt_utils import KeyDataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    RobertaTokenizer,
    RobertaTokenizerFast,
    RobertaForMaskedLM,
    pipeline,
)


class Bert:
    def __init__(self, model_checkpoint="roberta-base"):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            model_checkpoint, add_prefix_space=False
        )
        self.model = RobertaForMaskedLM.from_pretrained(model_checkpoint)


if __name__ == "__main__":
    bert = Bert()
    dataset = load_dataset("myradeng/cs230-news-unfiltered")
    dataset = masking_utils.read_eval_data(dataset, True)

    for idx, row in dataset.iterrows():
        output = masking_utils.mask_single_gender(
            bert.tokenizer, input_text=row["content"]
        )
        dataset.loc[idx, "content"] = output[0]
        dataset.loc[idx, "label"] = output[1]
    dataset = Dataset.from_pandas(dataset)
