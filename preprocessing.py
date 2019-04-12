from copy import copy
import logging
from numpy import array
from numpy import asarray
from numpy import zeros

"""
potential problem: running an old model may fail due to mappings
    being regenerated every time.
    TODO: fix mappings to a specified map by storing them in pkl file.
"""

def load_dataset(datasets, do_pad_words, embedding_size=None):
    """
    if pad_words, then each word in every dataset will be padded to the length of the longest word. PAD token is integer 0.
    All fields would be padded, which include 'tokens', 'raw_tokens', and 'boundaries'. This makes the training take 75s per epoch on just LSTM (~2x longer).
    """
    embeddings = []
    mappings = {}
    data = {}
    word_length = -1

    for datasetName, dataset in datasets.items():
        dataset_columns = dataset['columns']

        trainData = 'celex-data/%s/train.txt' % datasetName 
        devData = 'celex-data/%s/dev.txt' % datasetName 
        testData = 'celex-data/%s/test.txt' % datasetName 
        paths = {'train_matrix': trainData, 'dev_matrix': devData, 'test_matrix': testData}

        logging.info(":: Transform " + datasetName + " dataset ::")
        mappings, vocab_size, n_class_labels = make_mappings(paths.values(), do_pad_words)
        data[datasetName] = process_data(paths, dataset_columns, dataset, mappings)
        if do_pad_words:
            data[datasetName], word_length = pad_words(data[datasetName])

    embeddings = load_embeddings(embedding_size, mappings, vocab_size, datasetName)
    return (embeddings, data, mappings, vocab_size, n_class_labels, word_length)

def pad_words(data):
    """
    Pad each word to the length of the longest word. Token for PAD is the integer 0.
    """

    # Find the length of the longest word in the dataset for padding purposes.
    max_len = 0
    tokens = set()
    phones = set()
    for mat in ['dev_matrix', 'train_matrix', 'test_matrix']:
        for word in data[mat]:
            if len(word['raw_tokens']) > max_len:
                max_len = len(word['raw_tokens'])
            for tok in word['tokens']:
                tokens.add(tok)
            for phone in word['raw_tokens']:
                phones.add(phone)

    # pad both 'tokens' with 0 and 'raw_tokens' with 'PAD'
    for mat in ['dev_matrix', 'train_matrix', 'test_matrix']:
        for word in data[mat]:
            word['raw_tokens'] += ['PAD' for _ in range(max_len - len(word['raw_tokens']))]
            word['tokens'] += [0 for _ in range(max_len - len(word['tokens']))]
            word['boundaries'] += [0 for _ in range(max_len - len(word['boundaries']))]
            assert(len(word['raw_tokens']) == max_len)
            assert(len(word['tokens']) == max_len)
            assert(len(word['boundaries']) == max_len)

    return data, max_len

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
        with open(path, 'r') as f:
            for line in f:
                line = line.split('\t') # (phone, boundary)
                if len(line) == 1:
                    continue
                all_phones.add(line[0])
                class_labels.add(line[1])

    mappings = {}
    for i, phone in enumerate(all_phones):
        mappings[phone] = i + 1 if pad_words else i # reserve 0 for padding

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
        entry = {'raw_tokens':[], 'boundaries': []}
        with open(path, 'r') as f:
            for line in f:
                line = line.strip('\n').split('\t')
                if len(line) == 1:
                    # TEMP: only include length > 1
                    if len(entry['raw_tokens']) > 1:
                        entries.append(copy(entry))
                    entry['raw_tokens'] = []
                    entry['boundaries'] = []
                    continue

                entry['raw_tokens'].append(line[0])
                entry['boundaries'].append(int(line[1]))
            
        data[name] = entries

        # add 'tokens' to data
        for i, entry in enumerate(data[name]):
            data[name][i]['tokens'] = [mappings[raw] for raw in entry['raw_tokens']]

    return data

def read_conll_single(f_name):
    words = []
    with open(f_name, 'r') as f:
        word = []
        for line in f:
            line = line.split()
            if len(line) == 0:
                words.append({ 'tokens' : copy(word) })
                word = []
                continue
            
            word.append(line[0])

    return words

def load_embeddings(embedding_size, mappings, vocab_size, datasetName):
    """
    GloVe.
    How to use with Keras: https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
    """
    if(embedding_size == None):
        return []
    elif(embedding_size == 50):
        f_name = 'glove-50d-vectors.txt'
    elif(embedding_size == 100):
        f_name = 'glove-100d-vectors.txt'
    elif(embedding_size == 200):
        f_name = 'glove-200d-vectors.txt'
    else:
        raise ValueError('pretrained GloVe embeddings must be either 50, 100 or 200 dimensional.')

    # which language's embeddings
    if datasetName == 'english':
        location = './embeddings/optimized_disc/'
    else:
        raise ValueError('pretrained GloVe embeddings only exist for english right now.')

    embeddings_index = {}
    with open(location + f_name, 'r') as f:
        for line in f:
            line = line.split()
            phone = line[0]
            coefs = asarray(line[1:], dtype='float32')
            embeddings_index[phone] = coefs
        
        print('Loaded %s phone vectors.' % len(embeddings_index))

    # add PAD token (0) to embeddings
    embeddings_index['0'] = zeros(embedding_size)

    # generate the final embedding matrix indexed by the token value.
    embedding_matrix = zeros((vocab_size, embedding_size))
    for phone, index in mappings.items():
        embedding_vector = embeddings_index.get(phone)
        embedding_matrix[index] = embedding_vector
    return embedding_matrix
