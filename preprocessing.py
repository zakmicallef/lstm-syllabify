from copy import copy
import logging

import pandas as pd
"""
potential problem: running an old model may fail due to mappings
    being regenerated every time.
    TODO: fix mappings to a specified map by storing them in pkl file.
"""


def read_file(path, length):
    if path.endswith('.csv'):
        return read_csv_single(path, length)
    elif path.endswith('.conll'):
        return read_conll_single(path, length)

def load_dataset_from_csv(dataset, dataset_name, do_pad_words):
    train_data = "data/%s/train.csv" % dataset_name
    dev_data = "data/%s/dev.csv" % dataset_name
    test_data = "data/%s/test.csv" % dataset_name

    paths = {
        "train_matrix": train_data,
        "dev_matrix": dev_data,
        "test_matrix": test_data,
    }

    mappings, vocab_size, n_class_labels = make_mappings_from_csv(paths, do_pad_words)

    data = process_data_raw_text(paths, mappings)
    if do_pad_words:
        data, word_length = pad_words(data)

    embeddings = []
    
    # currently do not have pre-trained phonetic embeddings.
    # returning embeddings = []. Embeddings mst be trained.
    return (embeddings, data, mappings, vocab_size, n_class_labels, word_length)


def load_dataset(dataset, dataset_name, do_pad_words):
    """
    if pad_words, then each word in every dataset will be padded to the length
    of the longest word. PAD token is integer 0. All fields would be padded,
    which include 'tokens', 'raw_tokens', and 'boundaries'. This makes the
    training take 75s per epoch on just LSTM (~2x longer).

    Returns:
        EMBEDDINGS (not used)
            - numpy.ndarray holding 300 dimensional embeddings (each
                numpy.ndarray) that are not normalized to 0-1.
            - there is not an explicit mapping built into the structure, so
                they must be associated with the mappings data structure
            - embeddings are for the word inputs. word (raw_tokens) -> tokens
                -> embedding
        DATA
            - raw_tokens are phones in DISC format
            shape:
            data = {
                'english': {
                    'train_matrix': [
                        {
                            'tokens': [int, int, ... , int],
                            'boundaries': [int, int, ... , int],
                            'raw_tokens':[str, str, ..., str]
                        }, ...
                    ]
                    'dev_matrix': same as train_matrix
                    'test_matrix': same as train_matrix
                }
            }

        MAPPINGS
            - dictionary that maps tokens to a unique integer
        VOCAB_SIZE
            - number of possible inputs to the NN.
            - Usually is the number of phones in the langage being used.
        N_CLASS_LABELS
            - number of possible types of syllable boundaries. 
            - Default is two: either boundary (1) or no boundary (0)
        WORD_LENGTH
            - length of the longest word in the dataset
    """
    embeddings = []
    mappings = {}
    data = {}
    word_length = -1

    dataset_columns = dataset["columns"]

    train_data = "data/%s/train.txt" % dataset_name
    dev_data = "data/%s/dev.txt" % dataset_name
    test_data = "data/%s/test.txt" % dataset_name
    paths = {
        "train_matrix": train_data,
        "dev_matrix": dev_data,
        "test_matrix": test_data,
    }

    logging.info(":: Transform " + dataset_name + " dataset ::")
    mappings, vocab_size, n_class_labels = make_mappings(paths.values(), do_pad_words)
    data = process_data(paths, dataset_columns, dataset, mappings)
    if do_pad_words:
        data, word_length = pad_words(data)

    # currently do not have pre-trained phonetic embeddings.
    # returning embeddings = []. Embeddings mst be trained.
    return (embeddings, data, mappings, vocab_size, n_class_labels, word_length)


def pad_words(data):
    """
    Pad each word to the length of the longest word. Token for PAD is the integer 0.
    """
    print(data["dev_matrix"])
    # Find the length of the longest word in the dataset for padding purposes.
    max_len = 0
    tokens = set()
    phones = set()
    for mat in ["dev_matrix", "train_matrix", "test_matrix"]:
        for word in data[mat]:
            if len(word["raw_tokens"]) > max_len:
                max_len = len(word["raw_tokens"])
            for tok in word["tokens"]:
                tokens.add(tok)
            for phone in word["raw_tokens"]:
                phones.add(phone)

    # pad both 'tokens' with 0 and 'raw_tokens' with 'PAD'
    for mat in ["dev_matrix", "train_matrix", "test_matrix"]:
        for word in data[mat]:
            word["raw_tokens"] += [
                "PAD" for _ in range(max_len - len(word["raw_tokens"]))
            ]
            word["tokens"] += [0 for _ in range(max_len - len(word["tokens"]))]
            word["boundaries"] += [0 for _ in range(max_len - len(word["boundaries"]))]
            assert len(word["raw_tokens"]) == max_len
            assert len(word["tokens"]) == max_len
            assert len(word["boundaries"]) == max_len

    return data, max_len

def make_mappings_from_csv(paths, pad_words):
    all_phones = set()
    class_labels = set()

    for name, path in paths.items():
        syllabified_df = pd.read_csv(path)
        syllabified_words = syllabified_df['syllable'].astype(str).values.tolist()
        for syllabified_word in syllabified_words:
            syllabified_word = syllabified_word.strip("b'").strip("'")
            # entry = {"raw_tokens": [], "boundaries": []}
            for pos, char in enumerate(syllabified_word):
                # if pos == 0:
                #     entry["boundaries"].append(0)
                #     entry["raw_tokens"].append(char)

                # elif char != '-':
                is_syllable = 0 if syllabified_word[pos-1] != '-' else 1

                    # entry["boundaries"].append(is_syllable)
                    # entry["raw_tokens"].append(char)
            
                all_phones.add(char)
                class_labels.add(is_syllable)

    mappings = {}
    for i, phone in enumerate(all_phones):
        mappings[phone] = i + 1 if pad_words else i  # reserve 0 for padding

    vocab_size = len(mappings) + 1 if pad_words else len(mappings)
    return mappings, vocab_size, len(class_labels)

def make_mappings(paths, pad_words):
    """
    creates a unique mapping from phone to integer.
    Args:
        paths (list<str>): file paths that hold all possible phones
    Returns:
        dict phone->int,
        int: vocab size (# phone inputs)
        int: number of class labels (possible boundary classifications)
    """
    all_phones = set()
    class_labels = set()

    for path in paths:
        with open(path, "r") as f:
            for line in f:
                line = line.split("\t")  # (phone, boundary)
                if len(line) == 1:
                    continue
                all_phones.add(line[0])
                class_labels.add(line[1])

    mappings = {}
    for i, phone in enumerate(all_phones):
        mappings[phone] = i + 1 if pad_words else i  # reserve 0 for padding

    vocab_size = len(mappings) + 1 if pad_words else len(mappings)
    return mappings, vocab_size, len(class_labels)


def process_data(paths, dataset_columns, dataset, mappings):
    """
    hardcoded for certain columns. Mst be changed by hand.
    TODO: leverage details from dataset_columns and dataset.
    """
    data = {}

    for name, path in paths.items():

        # add 'raw_tokens' and 'boundaries' to data
        entries = []
        entry = {"raw_tokens": [], "boundaries": []}
        with open(path, "r") as f:
            for line in f:
                line = line.strip("\n").split("\t")
                if len(line) == 1:
                    # TEMP: only include length > 1
                    if len(entry["raw_tokens"]) > 1:
                        entries.append(copy(entry))
                    entry["raw_tokens"] = []
                    entry["boundaries"] = []
                    continue

                entry["raw_tokens"].append(line[0])
                entry["boundaries"].append(int(line[1]))

        data[name] = entries

        # add 'tokens' to data
        for i, entry in enumerate(data[name]):
            data[name][i]["tokens"] = [mappings[raw] for raw in entry["raw_tokens"]]

    return data

def process_data_raw_text(paths, mappings):
    data = {}

    for name, path in paths.items():
        entries = []

        syllabified_df = pd.read_csv(path)
        syllabified_words = syllabified_df['syllable'].astype(str).values.tolist()
        for syllabified_word in syllabified_words:
            syllabified_word = syllabified_word.strip("b'").strip("'")
            entry = {"raw_tokens": [], "boundaries": []}
            for pos, char in enumerate(syllabified_word):
                if pos == 0:
                    entry["boundaries"].append(0)
                    entry["raw_tokens"].append(char)

                elif char != '-':
                    is_syllable = 0 if syllabified_word[pos-1] != '-' else 1

                    entry["boundaries"].append(is_syllable)
                    entry["raw_tokens"].append(char)

            entries.append(entry)
        data[name] = entries

        for i, entry in enumerate(data[name]):
            data[name][i]["tokens"] = [mappings[raw] for raw in entry["raw_tokens"]]

    return data


def read_conll_single(f_name, final_length=28):
    # The length is hardcoded for English. Won't work with other languages...
    words = []
    current_length = 0
    warn_cropping_words = False
    with open(f_name, "r") as f:
        word = []
        for line in f:
            line = line.split()
            if len(line) == 0:
                word, is_longer = prep_word(word, final_length)
                if is_longer: warn_cropping_words = True
                words.append({"tokens": copy(word)})
                word = []
                current_length = 0
                continue
            current_length += 1
            word.append(line[0])
    if warn_cropping_words: logging.warning('Found words that are longer that model can fit (cropped words)')
    return words

def read_csv_single(f_name, final_length=28):
    word_df = pd.read_csv(f_name)
    warn_cropping_words = False
    words = []
    f = word_df['token'].astype(str).values.tolist()
    for word in f:
        word = list(word)
        word, is_longer = prep_word(word, final_length)
        if is_longer: warn_cropping_words = True
        words.append({"tokens": copy(word)})

    if warn_cropping_words: logging.warning('Found words that are longer that model can fit (cropped words)')
    return words

'''
word: list of letters
length: desired length
'''
def prep_word(word, length):
    is_longer = False
    word_length = len(word)
    if (word_length > length):
        is_longer = True
        word = word[:length]
    else:
        word += ["PAD" for x in range(length - word_length)]

    return word, is_longer


def create_data_matrix(words, mappings):
    # TODO: this should be merged with process_data
    data = []
    for word in words:
        token_transform_lst = []
        for char in word["tokens"]:
            if char in mappings:
                token_transform_lst.append(mappings[char])
            else:
                token_transform_lst.append("0")
        data.append({"raw_tokens": word["tokens"], "tokens": token_transform_lst})

    return data