from utils_data import create_maps, word_seq_lengths, char_seq_lengths
import nltk

# nltk.download('brown')
# nltk.download('universal_tagset')
data = nltk.corpus.brown.tagged_sents(categories='editorial', tagset='universal')

word2in, char2in, tag2in = create_maps(data)
PADDED_WORD_LEN = max(word_seq_lengths(data))
PADDED_CHAR_LEN = max(char_seq_lengths(data))
WORD_EMBED_DIM = 100
WORD_HIDDEN_DIM = 200
CHAR_EMB_DIM = 100
CHAR_HIDDEN_DIM = 100
EPOCH_NUM = 20

def all_constants():

    global PADDED_WORD_LEN, PADDED_CHAR_LEN, WORD_EMBED_DIM, WORD_HIDDEN_DIM, word2in, char2in, tag2in

    return word2in, char2in, tag2in, PADDED_WORD_LEN, PADDED_CHAR_LEN, WORD_EMBED_DIM, WORD_HIDDEN_DIM

def create_dataset_inputs(model_type):

    global PADDED_WORD_LEN, PADDED_CHAR_LEN, WORD_EMBED_DIM, WORD_HIDDEN_DIM, word2in, char2in, tag2in

    return PADDED_WORD_LEN, PADDED_CHAR_LEN, word2in, char2in, tag2in, model_type

def create_model_inputs(model_type, char_emb_dim=CHAR_EMB_DIM, char_hidden_dim=CHAR_HIDDEN_DIM, word_emb_dim=WORD_EMBED_DIM, word_hidden_dim=WORD_HIDDEN_DIM, num_layers=1,  linear_dim=500, dropout=0.5, bias=False):

    global word2in, char2in, tag2in

    # model_type can be 'main', 'dual' or 'vanilla' or 'dual-highway'
    if model_type == 'crf':
        return len(tag2in), len(char2in), char_emb_dim, char_hidden_dim, word2in, word_emb_dim, word_hidden_dim

    elif model_type == 'dual':
        return word_hidden_dim, len(tag2in), word2in, char2in, False, num_layers, word_emb_dim, char_hidden_dim, char_emb_dim, dropout, bias

    elif model_type == 'dual-highway':
        return word_hidden_dim, len(tag2in), word2in, char2in, True, num_layers, word_emb_dim, 150, char_emb_dim, dropout, bias

    elif model_type == 'vanilla':
        return word_hidden_dim, len(tag2in), word2in, linear_dim, 2, word_emb_dim, dropout, bias

    else:
        raise NameError('Incorrect model type name')
