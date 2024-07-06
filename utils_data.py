from collections import Counter
import torch
import torchtext
torchtext.disable_torchtext_deprecation_warning()
import torchtext.vocab

"""
1. Choose only words not punctuation nor number
2. Take only lowcase char
3. Glove doesn't have words like this
    1. georgia's it's jefferson's people's russia's
    2. tech. senate' america' hours' c.a.i.p.
    3. sycophantically brookmont stolzenbach -> unknown words
    4. i've i'm i'll they've we'll  shouldn't can't wouldn't aren't don't hasn't  wasn't
4. Min freq = 1 everywhere on this project
6. Trash data with lenghts < 2

https://pytorch.org/text/stable/vocab.html#glove

"""
def word_seq_lengths(data):
    """ Returns list of sentense lengths of words in data
        Including <end> tag """

    def _find_len(sen):
        a = [word for word, _ in sen if word[0].isalpha()]
        return len(a) + 1

    return [_find_len(sen) for sen in data]

def char_seq_lengths(data):
    """ Returns list of setense lengths of character in data"""

    def _find_len(sen):
        a = ' '.join(word for word, _ in sen if word[0].isalpha())
        return len(a) + 1

    return [_find_len(sen) for sen in data]
def create_maps(data, min_word_freq=1):
    """
    For ALL Data
    Creates word, char, tag maps.

    :param data: all data from nltk
    :param min_word_freq: words that occur fewer times than this threshold are binned as <unk>s
    :return: word, char, tag maps
    """
    def _change_word(word):
        if word[-2:] == '\'s':
            return word[:-2].lower()
        if not word[-1].isalpha():
            return word[:-1].lower()
        return word.lower()

    word_freq = Counter(_change_word(word) for sample in data for word, _ in sample if word[0].isalpha())

    word_map = {k: v + 1 for v, k in enumerate([w for w in word_freq.keys() if word_freq[w] >= min_word_freq])}
    char_map = {k: v + 1 for v, k in enumerate([c for c in (map(chr, range(97, 123)))])}
    tag_map = {k: v + 1 for v, k in enumerate(set(tag for sample in data for word, tag in sample if word[0].isalpha()))}

    word_map['<pad>'] = 0
    word_map['<end>'] = len(word_map)
    word_map['<unk>'] = len(word_map)
    char_map['<pad>'] = 0
    char_map['<end>'] = len(char_map)
    tag_map['<pad>'] = 0
    tag_map['<start>'] = len(tag_map)
    tag_map['<end>'] = len(tag_map)

    return word_map, char_map, tag_map
def create_input_words(data, idx, word_map, max_len, pad_idx=0):
    """
    For one sentence in data create padded tensor of words for model input
    :param data: train/valid/test data
    :param idx:
    :param word_map: map from word to idx
    :param max_len: the longest sentence in data
    :return: tensor of words indexes, true sentence length
    """

    words = [word_map.get(word.lower(), word_map['<unk>']) for word, _ in data[idx] if word[0].isalpha()]
    words.append(word_map['<end>'])
    sen_len = len(words)
    words += [pad_idx] * (max_len - len(words))

    return torch.LongTensor(words), torch.tensor(sen_len)

def from_tensor_to_words(tensor, word_map):

    """"
    :return word sentence for human reading
    """

    index_to_word = [word for word, _ in sorted(word_map.items(), key=lambda x: x[1])]
    return [index_to_word[idx] for idx in tensor.tolist()]

def create_input_tag(data, idx, tag_map, max_len):

    """
    https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Sequence-Labeling?tab=readme-ov-file#tags
    Returns input padded tags for CRF model and for other models

    """
    pad_idx = tag_map['<pad>']
    prev_tag = '<start>'
    crf_tags = []
    for word, tag in data[idx]:
        if word[0].isalpha():
            a = tag_map[prev_tag] * len(tag_map) + tag_map[tag]
            crf_tags .append(a)
            prev_tag = tag

    crf_tags.append(tag_map[prev_tag] * len(tag_map) + tag_map['<end>'])
    correct_tags = [t % len(tag_map) for t in crf_tags]

    sen_len = len(correct_tags)

    crf_tags += [pad_idx] * (max_len - len(crf_tags))
    correct_tags += [pad_idx] * (max_len - len(correct_tags))

    return torch.LongTensor(crf_tags), torch.LongTensor(correct_tags), torch.tensor(sen_len)

def from_tensor_to_tag(tensor, tag_map):
    """"
    :return tag sentence for human reading

    """
    index_to_tag = [tag for tag, _ in sorted(tag_map.items(), key=lambda x: x[1])]
    return [index_to_tag[idx] for idx in tensor.tolist()]

def create_input_char(data, idx, char_map, max_len, pad_idx=0):
    """
    Returns  padded encoded forward / backward chars,
    padded forward / backward character markers:
    positions of spaces and <end> character as [0, 0, 1, 0, 0, 0, 0, 1 ....
    Words are predicted or encoded at these places in the language and tagging models respectively
    """

    a = ' '.join(word.lower() for word, _ in data[idx] if word[0].isalpha())

    char_stream = [char for char in a if char.isalpha()]

    char_maps_f = [char_map[char] for char in char_stream]
    char_maps_b = [char_map[char] for char in char_stream[::-1]]

    char_maps_f.append(char_map['<end>'])
    char_maps_b.append(char_map['<end>'])

    char_maps_f += [pad_idx] * (max_len - len(char_maps_f))
    char_maps_b += [pad_idx] * (max_len - len(char_maps_b))

    char_makers_f = []
    for i in range(len(a) - 1):
        if a[i + 1].isspace():
            char_makers_f.append(1)
        elif a[i].isspace():
            continue
        else:
            char_makers_f.append(0)

    char_makers_b = char_makers_f[::-1]

    char_makers_f.append(1) # the last charecter
    char_makers_f.append(1) # the <end> tag
    char_makers_b.append(1)
    char_makers_b.append(1)

    sen_len = sum(char_makers_f)

    char_makers_f += [pad_idx] * (max_len - len(char_makers_f))
    char_makers_b += [pad_idx] * (max_len - len(char_makers_b))


    return torch.LongTensor(char_maps_f), torch.LongTensor(char_maps_b), \
           torch.LongTensor(char_makers_f), torch.LongTensor(char_makers_b), torch.tensor(sen_len)

def load_embeddings(word_map, min_freq=1, EMBEDDING_DIM=100):
    """
    Copy of function in models to research Glove vectors
    https://pytorch.org/text/stable/vocab.html#id1
    :param word_map:
    :param min_freq:
    :return: vocab that contains pretrained embeddings and number of words that aren't pretrained
    """
    glove = torchtext.vocab.GloVe(name='6B', dim=EMBEDDING_DIM)
    my_vocab = torchtext.vocab.vocab(word_map, min_freq=min_freq, specials=['<pad>'], )
    my_vocab.vectors = glove.get_vecs_by_tokens(my_vocab.get_itos())

    # Rather than initializing new token embeddings to 0's, use torch.randn to create some diversity
    # between the different token embeddings that weren't preloaded with GloVe
    # but keep 0's for <pad> token
    out_of_corpus = []  # out-of-corpus words
    for i in range(my_vocab.vectors.shape[0]):
        if my_vocab.vectors[i].equal(torch.zeros(100)):
            my_vocab.vectors[i] = torch.randn(100) if my_vocab.get_itos()[i] != '<pad>' else my_vocab.vectors[i]
            out_of_corpus.append(my_vocab.lookup_token(i))

    return my_vocab, out_of_corpus


