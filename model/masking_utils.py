def mask_single_gender(tokenizer, gender_identifiers=[], input_text=""):
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
    matches = list(re.finditer(regex, input_text))

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
        input_text[:single_match_start]
        + tokenizer.mask_token
        + input_text[single_match_end:]
    )
    return input_text, label


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
    regex = re.compile(r"\b(?:%s)\b" % "|".join(gender_identifiers))
    return regex.sub(tokenizer.mask_token, input_text)


def split_to_contexts(eval_dataset, context_size=100):
    concat_text = " ".join(eval_dataset)
    words = concat_text.split()
    grouped_words = [
        " ".join(words[i : i + context_size])
        for i in range(0, len(words), context_size)
    ]
    return grouped_words


def read_eval_data(dataset, downsample=False):
    eval_dataset = dataset["test"]
    # Downsample if running on colab
    if downsample:
        downsampled_dataset = dataset["test"].train_test_split(test_size=100, seed=42)
        eval_dataset = downsampled_dataset["test"]
    repartitioned = split_to_contexts(eval_dataset[CONTENT_ROW])
    eval_dataset_df = pd.DataFrame({"content": repartitioned})
    return eval_dataset_df


def compute_single_prob(predictions):
    woman_prob_numerator = 0
    man_prob_numerator = 0
    all_gender_denominator = 0
    for prediction in predictions:
        token_string = prediction[TOKEN_STRING].strip()
        if token_string in WOMAN_KEYWORDS:
            woman_prob_numerator += prediction[SCORE]
            all_gender_denominator += prediction[SCORE]
        if token_string in MAN_KEYWORDS:
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


def compute_probs(predictions):
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
            woman_prob, man_prob = compute_single_prob(prediction)
            woman_prob_list.append(woman_prob)
            man_prob_list.append(man_prob)
        woman_prob = np.mean(woman_prob_list)
        man_prob = np.mean(man_prob_list)
    else:
        woman_prob, man_prob = compute_single_prob(predictions)
    return woman_prob, man_prob


def evaluate(eval_df):
    model_fn = pipeline("fill-mask", model="roberta-base")
    predictions = []
    woman_probs = []
    man_probs = []

    for prediction in tqdm(model_fn(KeyDataset(eval_df, "content"), top_k=TOP_K)):
        (woman_prob, man_prob) = compute_probs(prediction)
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
