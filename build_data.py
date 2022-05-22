import argparse
from preprocess_data import *
import os
from string import punctuation
import numpy as np
import pandas as pd
import re
from collections import Counter

woman_keywords = ["woman", "women", "female", "she", "her", "hers"]
man_keywords = ["man", "men", "male", "he", "his", "him"]
CONTENT_COLUMN_NAME = "content"


def shuffle_data(df, seed=0):
    return df.sample(frac=1, random_state=0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process input files into train, dev, test"
    )
    parser.add_argument("-i", dest="input_uri", type=str)
    parser.add_argument("-o", dest="output_dir", type=str)
    # Whether to filter for only sentences with a gender pronoun present
    parser.add_argument(
        "--filter_gender", default=False, action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--proportion_gender", default=False, action=argparse.BooleanOptionalAction
    )
    # This is a hyperparameter used to tune how many rows to drop in order to achieve roughly ~50/50 pronoun balance
    parser.add_argument("-r", dest="ratio_param", type=str, default=1.35)
    return parser


def filter_gender_entries(data):
    gender_keyword_filtered_df = data[
        data[CONTENT_COLUMN_NAME + "_list"].apply(
            lambda x: any([k in x for k in woman_keywords])
        )
    ]
    gender_keyword_filtered_df = gender_keyword_filtered_df[
        gender_keyword_filtered_df[CONTENT_COLUMN_NAME + "_list"].apply(
            lambda x: any([k in x for k in man_keywords])
        )
    ]
    return gender_keyword_filtered_df


# TODO: Make this more systematic rather than an approximation
def proportion_gender_entries(data, male_pronoun_count, female_pronoun_count, ratio):
    women = data[
        data["content_list"].apply(lambda x: any([k in x for k in woman_keywords]))
    ]
    men = data[
        data["content_list"].apply(lambda x: any([k in x for k in man_keywords]))
    ]
    men_with_women = men[
        men["content_list"].apply(lambda x: any([k in x for k in woman_keywords]))
    ]
    men_no_women = men[
        men["content_list"].apply(
            lambda x: any([k in x for k in woman_keywords]) == False
        )
    ]

    remove_n = int((male_pronoun_count - female_pronoun_count) / ratio)
    drop_indices = np.random.choice(men_no_women.index, remove_n, replace=False)
    men_no_women_subset = men_no_women.drop(drop_indices)
    rebalanced = pd.concat([women, men_no_women_subset], axis=0)
    return rebalanced


def count_gender_entries(data):
    series = pd.Series(
        Counter([y for x in data[CONTENT_COLUMN_NAME + "_list"] for y in x])
    )
    female_pronoun_count = 0
    male_pronoun_count = 0
    for female_pronoun in woman_keywords:
        female_pronoun_count += series[female_pronoun]
    for male_pronoun in man_keywords:
        male_pronoun_count += series[male_pronoun]

    return female_pronoun_count, male_pronoun_count


def main():
    parser = parse_args()
    args = parser.parse_args()
    uri = args.input_uri
    output_dir = args.output_dir
    filter_gender = args.filter_gender
    proportion_gender = args.proportion_gender
    ratio = float(args.ratio_param)

    data = read_from_uris(uri)

    # Basic cleaning: drop unnecessary rows, remove extra spaces, add dummy label
    data = data.drop(columns=["Unnamed: 0"])
    data[CONTENT_COLUMN_NAME] = data[CONTENT_COLUMN_NAME].str.replace(" +", " ")
    data["label"] = 0

    # Split data entries into separate sentences and remove empty strings
    data = split_df_list(data, CONTENT_COLUMN_NAME, ".")
    data = data[data["content"].str.len() > 0]
    data["content"] = np.where(
        data["content"].str.endswith("."), data["content"], data["content"] + "."
    )
    data["content"] = data["content"].str.strip()
    data["content"] = data["content"].astype(str)
    data["content"] = data["content"].apply(lambda x: contractions.fix(x))

    data[CONTENT_COLUMN_NAME + "_list"] = data[CONTENT_COLUMN_NAME].str.split()

    # (Configurable) Filter for single instance pronoun entries
    if filter_gender:
        data["pronoun_count"] = data.apply(
            lambda row: sum(
                [item in row["content_list"] for item in woman_keywords + man_keywords]
            ),
            axis=1,
        )
        data = data[data["pronoun_count"] == 1]

    # Count initial gender proportions
    female_pronoun_count, male_pronoun_count = count_gender_entries(data)
    print("Male pronoun count: " + str(male_pronoun_count))
    print("Female pronoun count: " + str(female_pronoun_count))

    # Ensure approx equal split between male / female gender pronouns in training data
    if proportion_gender:
        data = proportion_gender_entries(
            data, male_pronoun_count, female_pronoun_count, ratio
        )
        female_pronoun_count, male_pronoun_count = count_gender_entries(data)
        print("Male pronoun count after rebalancing: " + str(male_pronoun_count))
        print("Female pronoun count after rebalancing: " + str(female_pronoun_count))

    # Shuffle and split data intro train, dev, test
    data = shuffle_data(data)
    split_1 = int(0.8 * len(data))
    split_2 = int(0.9 * len(data))
    train_data = data[:split_1]
    dev_data = data[split_1:split_2]
    test_data = data[split_2:]

    # Write data out to three separate files under output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Created new directory at: " + output_dir)
    print("Writing data to: " + output_dir)
    train_data.to_csv(output_dir + "/train.csv")
    dev_data.to_csv(output_dir + "/dev.csv")
    test_data.to_csv(output_dir + "/test.csv")


if __name__ == "__main__":
    main()
