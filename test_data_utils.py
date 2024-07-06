import torch
from model_inputs import *
from dataset import *
from utils_data import *

torch.manual_seed(8)

glove = torchtext.vocab.GloVe(name='6B', dim=100)
# word2in, char2in, tag2in, PADDED_WORD_LEN, PADDED_CHAR_LEN, _, _= all_constants()
print(tag2in)

def test_char_data_utils(data, char2in, word2in, idx=0):

    char_stream_f, char_stream_b, char_maker_f, char_maker_b, l = create_input_char(data, idx, char2in, PADDED_CHAR_LEN)
    words, l = create_input_words(data, idx, word2in, PADDED_WORD_LEN)
    # print(sorted(char2in.items(), key=lambda x: x[1]))
    print('^'*50)
    print(char_stream_f)
    print('^' * 50)
    # print(char_stream_f.tolist()[:130])
    # print(char_stream_b.tolist()[:130])
    print('^'*50)
    print(char_maker_f.tolist()[:100])
    print('^' * 50)
    print(words[:20])
    print(f'Number of words {torch.nonzero(words).size(0)}')
    print(f'Number of ones in character makers {torch.sum(char_maker_f)}')
    print(f'The <end> tag on {l - 1} place, {words[l - 1]}')

# test_char_data_utils(data, char2in, word2in)

def research_data(data):

    """
        Check if sizes of tag and word sentence are same
        Check if all sen have same number of 0
        Check if <end> tag at the same place in tags and words

    """
    word_map, char_map, tag_map = word2in, char2in, tag2in
    for i, sen in enumerate(data):

        w_sen, w_l = create_input_words(data, i, word_map, PADDED_WORD_LEN)
        t_sen, true_tag, t_l = create_input_tag(data, i, tag_map, PADDED_WORD_LEN)
        w_idx, t_idx = word_map['<end>'], tag_map['<end>']

        if len(w_sen) != len(true_tag):
            print(f"Found mismatched padded sentence lengths {i}")
            print(sen)
            print(w_sen)
            print(true_tag)
            break
        elif w_sen.count_nonzero() != true_tag.count_nonzero() != w_l != t_l:
            print(f"Found mismatched sentence lengths {i}")
            print(sen)
            print(w_sen)
            print(true_tag)
            break
        elif w_sen[w_l - 1] != w_idx or true_tag[t_l - 1] != t_idx:
            print(f"Found mismatched in <end> tag {i}")
            print(sen)
            print(w_sen)
            print(true_tag)
            print(w_l, t_l)
            break
    else:
        print('Everything works well;)')
#
# research_data(data)

def test_tags_data():

    global PADDED_WORD_LEN, PADDED_CHAR_LEN, word2in, char2in, tag2in,  word_sen_len

    tags, true_tags, l = create_input_tag(data, 0, tag2in, PADDED_WORD_LEN)
    print(data[0])
    print(true_tags)
    print(tags)
    print(l)
    print(from_tensor_to_tag(true_tags, tag2in))


test_tags_data()

def test_embeds(inx, data):

    print('Original sentence: \n, \n', ' '.join(word for word, _ in data[inx]))
    print('*' * 100)

    word2in, _, _ = create_maps(data)
    pad_len = 82
    sentence, sen_l = create_input_words(data, inx, word2in, pad_len)

    print('Input to a model: \n, \n', sentence.tolist())
    print('*' * 100)
    print(f'Number of words in sentence: {sen_l}')

    vocab, out_of_corpus = load_embeddings(word2in)
    s = ' '.join([vocab.lookup_token(i) for i in sentence])

    print('Sentence loaded to embeddings and turned back: \n, \n', s)
    print('*' * 100)
    print(f'Number of out-of-corpus words: {len(out_of_corpus)} out of {len(word2in)}')
    print('Out of corpus words ', *out_of_corpus)
    # print(f"Example of embedded word \"{data[inx][1][0]}\": \n {vocab.vectors[vocab.get_stoi()[data[inx][1][0]]]}")
    # print(type(vocab.vectors[vocab.get_stoi()[data[inx][1][0]]]))

# test_embeds(0, data)

def test_dataset(data):

    d_i = create_dataset_inputs('vanilla')
    dataset = POSDataset(data, *d_i)
    print(dataset[0][0])
    print(dataset[0][1])
    print(dataset[0][2])
    print(dataset[0][3])

# test_dataset(data)

def toy_example_decoder():

    scores, targets = torch.randint(-10, 10, (2, 7, 6, 6)), torch.tensor([[11, 8, 18, 19, 0, 0], [22, 13, 19, 0, 0, 0]])
    lengths, real_targets = torch.tensor([4, 3]), torch.tensor([[1, 3, 3, 4, 0, 0, 0], [2, 3, 4, 0, 0, 0, 0]])
    # I want be <end> ==> <start>, pronoun, verb, verb, <end>, <pad>, pad
    # Cat sees <end> ==> <start>, NOUN, VERB, <end>, <pad>, pad, pad
    # I have 5 tags: NOUN, PRONOUN, VERB, <end>, <start>
    tag_map = {'NOUN': 2, 'PRONOUN': 1, 'VERB': 3, '<start>': 4, '<end>': 5, '<pad>': 0}
    batch_size = 2
    word_pad_len = 7
    tag_set_size = 6

    def _viterbi_decoder(scores, lengths, tag_set_size, batch_size):
        """
        https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Sequence-Labeling?tab=readme-ov-file#viterbi-decoding
        """
        res = []
        for i in range(batch_size):
            viterbi_score = torch.zeros((tag_set_size, lengths[i]))
            backpointer = torch.zeros((tag_set_size), dtype=torch.long)
            viterbi_score[:, 0] = scores[i, 0, 0, :]
            for t in range(1, lengths[i]):
                backpointer[:] = viterbi_score[:, t - 1]
                max_scores, _ = torch.max(backpointer.unsqueeze(1).repeat(1, tag_set_size) + scores[i, t, :, :], dim=0)
                viterbi_score[:, t] = max_scores
            print(f"For sentence {i + 1}")
            print(viterbi_score.tolist())
            _, a = torch.max(viterbi_score[1:-1, :], dim=0)
            a = torch.add(a, 1)
            a[lengths[i] - 1] = 5
            print(a)
            res.append(a)
        res = torch.stack(
            [torch.cat((res[i], torch.zeros((word_pad_len - lengths[i]))), dim=0) for i in range(batch_size)]
        )
        print(res.tolist())

    _viterbi_decoder(scores, lengths, tag_set_size, batch_size)

