import torch
import torch.nn as nn

import torchtext
torchtext.disable_torchtext_deprecation_warning()
import torchtext.vocab

torch.manual_seed(1608)

EMBEDDING_DIM = 100

def _load_embeddings(word_map, min_freq=1):
    """
     Preload GloVe embeddings into vocabulary
    :param word_map:
    :param min_freq:
    :return: vocab that contains pretrained embeddings and number of words that aren't pretrained
    """
    glove = torchtext.vocab.GloVe(name='6B', dim=EMBEDDING_DIM)
    my_vocab = torchtext.vocab.vocab(word_map, min_freq=min_freq, specials=['<pad>'])
    my_vocab.vectors = glove.get_vecs_by_tokens(my_vocab.get_itos())

    # Rather than initializing new token embeddings to 0's, use torch.randn to create some diversity
    # between the different token embeddings that weren't preloaded with GloVe
    # but keep 0's for <pad> token
    for i in range(my_vocab.vectors.shape[0]):
        if my_vocab.vectors[i].equal(torch.zeros(100)):
            my_vocab.vectors[i] = torch.randn(100) if my_vocab.get_itos()[i] != '<pad>' else my_vocab.vectors[i]

    return my_vocab

class VanillaPOSTagger(nn.Module):

    def __init__(self, hidden_dim, output_dim, word2in, linear_dim=500, num_layers=2, embedding_dim=EMBEDDING_DIM, dropout=0.5, bias=False):
        """
        :param hidden_dim: word hidden dimension for LSTM
        :param output_dim: number of tags
        :param word2in: word to int map
        :param linear_dim: fully connected layers hidden dimension
        :param num_layers: number of layers in LSTM
        :param embedding_dim: word embedding dimension
        :param dropout: dropout probability
        :param bias: False to not let signal from padding

        """

        super(VanillaPOSTagger, self).__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        my_vocab = _load_embeddings(word2in)
        vocab_size, emb_size = my_vocab.vectors.shape

        self.embedding = nn.Embedding(vocab_size, embedding_dim, _weight=my_vocab.vectors)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True, bias=bias)
        self.fc = nn.Sequential(nn.Linear(hidden_dim * 2, linear_dim), nn.ReLU(), nn.Linear(linear_dim, output_dim))
        self.activation = nn.LogSoftmax(dim=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, word_seq, tags, lengths):
        """
        :param  word_seq,: word padded sequence
        :param tags: tags padded sequence
        :param lengths: true lengths of sentence
        :return:  predicted scores for all tags, tags sequence, true lengths of sentence

        """
        # sentence = [batch size, sent len]
        embedded = self.dropout(self.embedding(word_seq)) #  [batch size, sent len, embed_dim]

        outputs, (hidden, cell) = self.lstm(embedded) # [batch size, sent len, hid dim * n directions]

        predictions = self.fc(self.dropout(outputs)) # [batch size, sent len, output dim]

        preds = self.activation(self.dropout(predictions))

        return preds, tags, lengths

class Highway(nn.Module):
    """
    Highway Network with one layer
    """

    def __init__(self, size):
        """
        :param size: size of linear layer (matches input size)
        """
        super(Highway, self).__init__()
        self.size = size
        self.transform = nn.Linear(size, size)
        self.gate = nn.Linear(size, size)

    def forward(self, x):
        """
        :param x: input tensor
        :return: output tensor, with same dimensions as input tensor
        """
        transformed = nn.functional.relu(self.transform(x))  # transform input
        g = nn.functional.sigmoid(self.gate(x))  # calculate how much of the transformed input to keep

        out = g * transformed + (1 - g) * x  # combine input and transformed input in this ratio

        return out

class DualPOSTagger(nn.Module):

    def __init__(self, word_hidden_dim, output_dim, word2in, char2in, highway, num_layers=1, word_embedding_dim=EMBEDDING_DIM, char_hidden_dim=100, char_embedding_dim=100, dropout=0.5, bias=False):
        """
        :param word_hidden_dim: word hidden dimension for LSTM
        :param output_dim: number of tags
        :param word2in: word to int map
        :param char2in: char to int map
        :param highway: 2 model types, one includes Highway Network one doesn't
        :param num_layers: number of layers in both LSTMs and
        :param word_embedding_dim: word embedding dimension
        :param char_hidden_dim: charecter embedding dimension
        :param dropout: dropout probability
        :param bias: False to not let signal from padding

        """

        super(DualPOSTagger, self).__init__()

        self.highway = highway

        my_vocab = _load_embeddings(word2in)
        vocab_size, emb_size = my_vocab.vectors.shape

        assert emb_size == word_embedding_dim, 'Embedding dim doesn\'t match in model'
        if not highway:
            assert char_hidden_dim == word_embedding_dim, 'Dims should match'

        self.word_embedding = nn.Embedding(vocab_size, word_embedding_dim, _weight=my_vocab.vectors)

        self.char_embedding = nn.Embedding(len(char2in), char_embedding_dim)
        self.char_lstm = nn.LSTM(char_embedding_dim, char_hidden_dim, batch_first=True, num_layers=num_layers, bias=bias)
        self.lstm = nn.LSTM(word_embedding_dim, word_hidden_dim, batch_first=True, num_layers=num_layers, bias=bias)
        self.lstm_hw = nn.LSTM(word_embedding_dim + char_hidden_dim, word_hidden_dim, batch_first=True, num_layers=num_layers, bias=bias)
        self.hw = Highway(char_hidden_dim)  # highway to transform combined forward char LSTM outputs for use in the word LSTM
        self.fc = nn.Linear(word_hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, word_seq, char_seq, char_makers, tags, lengths):
        """

        :param word_seq: word padded sequence
        :param char_seq: character padded sequence
        :param char_makers: list contains 0 and 1 where 1 in places where words are ending
        :param tags: tags padded sequence
        :param lengths: true lengths of sentence
        :return: predicted scores for all tags, tags sequence, true lengths of sentence
        """

        embeds = self.word_embedding(word_seq)  # [batch size, sent len, word_embed]
        char_input = self.char_embedding(char_seq)  # [batch, seq len, char_embed]
        char_out, _ = self.char_lstm(char_input)  # [batch, seq_len, char_hid_dim]
        char_out = char_makers.unsqueeze(2) * char_out  # to take output only from the end of words

        # https://discuss.pytorch.org/t/how-can-i-move-all-zeroes-to-end-of-array/43092/4

        char_out = torch.stack(
            [torch.cat((char_out[i][~(char_out[i].sum(dim=1) == 0)], char_out[i][(char_out[i].sum(dim=1) == 0)]), dim=0)
             for i in range(char_out.shape[0])]
        )  # move all 0 rows to bottom

        assert torch.sum(char_out[0, 200:, :]) == 0, 'Zeros not moved to bottom'

        char_out = char_out[:, :embeds.size()[1], :]  # [batch_size, sent_len, char_hid_dim]

        if not self.highway:
            combined = torch.add(embeds, char_out)  # [batch, seq_len, word_embed_dim]
            words_out, _ = self.lstm(combined)
            preds = self.fc(words_out)  # [batch, seq_len, tag_len]

            return preds, tags, lengths
        else:
            subword = self.hw(char_out)
            combined = torch.cat((embeds, subword), dim=2)  # [batch, seq_len, embed_dim + char_hid_dim]
            words_out, _ = self.lstm_hw(combined)
            preds = self.fc(self.dropout(words_out))  # [batch, seq_len, tag_len]

            return preds, tags, lengths

class CRF(nn.Module):

    def __init__(self, hidden_dim, tagset_size):
        """
        :param hidden_dim: size of word RNN/BLSTM's output
        :param tagset_size: number of tags
        """
        super(CRF, self).__init__()
        self.tagset_size = tagset_size
        self.emission = nn.Linear(hidden_dim, self.tagset_size)
        self.transition = nn.Parameter(torch.Tensor(self.tagset_size, self.tagset_size)) # A kind of Tensor that is to be considered a module parameter.
        self.transition.data.zero_() # set grads to 0

    def forward(self, feats):
        """
        :param feats: output of word RNN/BLSTM, a tensor of dimensions (batch_size, timesteps, hidden_dim)
        :return: CRF scores, a tensor of dimensions (batch_size, timesteps, tagset_size, tagset_size)
        """
        self.batch_size = feats.size(0)
        self.timesteps = feats.size(1)

        emission_scores = self.emission(feats)  # [batch_size, timesteps, tagset_size] # For a sentence of length L, emission scores would be an L, m tensor
        # Since the emission scores at each word do not depend on the tag of the previous word, we create a new dimension like L, _, m
        # and broadcast (copy) the tensor along this direction to get an L, m, m tensor.
        emission_scores = emission_scores.unsqueeze(2).expand(self.batch_size, self.timesteps, self.tagset_size,
                                                              self.tagset_size)  # [batch_size, timesteps, tagset_size, tagset_size]


        # The transition scores are an m, m tensor. Since the transition scores are global and do not depend on the word, we create a new dimension like _, m, m
        # and broadcast (copy) the tensor along this direction to get an L, m, m tensor
        crf_scores = emission_scores + self.transition.unsqueeze(0).unsqueeze(0)  # [batch_size, timesteps, tagset_size, tagset_size]

        return crf_scores

class LSTMTagger(nn.Module):
    """
    Main Model
    """

    def __init__(self, tagset_size, charset_size, char_emb_dim, char_hidden_dim, word2in,
                  word_emb_dim, word_hidden_dim, char_rnn_layers=1, bias=False, word_rnn_layers=1, dropout=0.5):
        """
        :param tagset_size: number of tags
        :param charset_size: size of character vocabulary 29
        :param char_emb_dim: size of character embeddings
        :param char_hidden_dim: size of character RNNs/LSTMs
        :param word2in: input vocabulary
        :param word_emb_dim: size of word embeddings 100
        :param word_hidden_dim: size of word RNN/BLSTM
        :param char_rnn_layers: number of layers in character RNNs/LSTMs 1
        :param word_rnn_layers:  number of layers in word RNNs/LSTMs 1
        :param dropout: dropout 0.5
        """

        super(LSTMTagger, self).__init__()

        self.tagset_size = tagset_size
        self.charset_size = charset_size
        self.char_emb_dim = char_emb_dim
        self.char_hidden_dim = char_hidden_dim
        self.char_rnn_layers = char_rnn_layers

        self.wordset_size = len(word2in)
        self.word_emb_dim = word_emb_dim
        self.word_hidden_dim = word_hidden_dim
        self.word_rnn_layers = word_rnn_layers

        my_vocab = _load_embeddings(word2in)
        vocab_size, emb_size = my_vocab.vectors.shape

        self.char_embeds = nn.Embedding(self.charset_size, self.char_emb_dim)
        self.forw_char_lstm = nn.LSTM(self.char_emb_dim, self.char_hidden_dim, num_layers=self.char_rnn_layers,
                                      bidirectional=False, batch_first=True)
        self.back_char_lstm = nn.LSTM(self.char_emb_dim, self.char_hidden_dim, num_layers=self.char_rnn_layers,
                                      bidirectional=False, batch_first=True)

        self.word_embeds = nn.Embedding(vocab_size, self.word_emb_dim, _weight=my_vocab.vectors)

        self.word_blstm = nn.LSTM(self.word_emb_dim + self.char_hidden_dim * 2, self.word_hidden_dim // 2,
                                  num_layers=self.word_rnn_layers, bidirectional=True)

        self.subword_hw = Highway(2 * self.char_hidden_dim)  # highway to transform combined forward and backward char LSTM outputs for use in the word BLSTM

        self.crf = CRF((self.word_hidden_dim // 2) * 2, self.tagset_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, wmap, cmaps_f, cmaps_b, cmakers_f, cmakers_b, tmaps, tmap_correct, lengths):
        """

        :param wmap: word padded sequence
        :param cmaps_f: forward character padded sequence
        :param cmaps_b: backward character padded sequence
        :param cmakers_f: padded forward character markers as [0, 0, 1, 0, 0, 1, 0 ....],
        :param cmakers_b: padded backward character markers as [0, 0, 1, 0, 0, 1, 0 ....],
        :param tmaps: input padded tags for CRF model
        :param tmap_correct: true padded tags
        :param lengths: true lengths of sentence
        :return: crf_scores, tmaps, lengths, tmap_correct
        """

        self.word_pad_len = wmap.size(1)

        cf = self.char_embeds(cmaps_f)  # [batch_size, char_pad_len, char_emb_dim]
        cb = self.char_embeds(cmaps_b)

        cf, _ = self.forw_char_lstm(cf)  # [batch_size, char_pad_len, char_hid_dim]
        cb, _ = self.back_char_lstm(cb)

        # Select RNN outputs only at marker points (spaces in the character sequence)
        cf = cmakers_f.unsqueeze(2) * cf
        cb = cmakers_b.unsqueeze(2) * cb
        cf_selected = torch.stack(
            [torch.cat((cf[i][~(cf[i].sum(dim=1) == 0)], cf[i][(cf[i].sum(dim=1) == 0)]), dim=0)
             for i in range(cf.shape[0])]
        )  # move all 0 rows to bottom
        cb_selected = torch.stack(
            [torch.cat((cb[i][~(cb[i].sum(dim=1) == 0)], cb[i][(cb[i].sum(dim=1) == 0)]), dim=0)
             for i in range(cb.shape[0])]
        )  # move all 0 rows to bottom

        cf_selected = cf_selected[:, :self.word_pad_len, :] # (batch_size, word_pad_len, char_hidden_dim)
        cb_selected = cb_selected[:, :self.word_pad_len, :]

        w = self.word_embeds(wmap)
        w = self.dropout(w)

        # Sub-word information at each word
        subword = self.subword_hw(self.dropout(
            torch.cat((cf_selected, cb_selected), dim=2)))  # (batch_size, word_pad_len, 2 * char_hidden_dim)

        w = torch.cat((w, subword), dim=2)  # (batch_size, word_pad_len, word_emb_dim + 2 * char_hidden_dim)

        w, _ = self.word_blstm(w) # [batch_size, word_pad_len,  word_emb_dim + 2 * char_hidden_dim]

        crf_scores = self.crf(w)  # (batch_size, word_pad_len, tagset_size, tagset_size)

        return crf_scores, tmaps, lengths, tmap_correct

class ViterbiLoss(nn.Module):
    """
    Viterbi Loss.
    """

    def __init__(self, tag_map):
        """
        :param tag_map: tag map
        """
        super(ViterbiLoss, self).__init__()

        self.start_tag = tag_map['<start>']
        self.end_tag = tag_map['<end>']
        self.tagset_size = len(tag_map)

    def forward(self, scores, targets, lengths, tmap_correct):
        """
        :param scores: CRF scores
        :param targets: true tags indices in unrolled CRF scores
        :return: viterbi loss
        """

        batch_size = scores.size(0)
        word_pad_len = scores.size(1)

        # Gold score
        targets = targets.unsqueeze(2)
        scores_at_targets = torch.gather(scores.view(batch_size, word_pad_len, -1), 2, targets).squeeze(2)  # (batch_size, word_pad_len)
        scores_at_targets = torch.stack(
            [torch.cat((scores_at_targets[i, :lengths[i]], torch.tensor([0] * (word_pad_len - lengths[i]))), dim=0)
                                for i in range(batch_size)]) # remove padding values
        gold_score = scores_at_targets.sum()


        sorted_lengths, word_sort_ind = lengths.sort(dim=0, descending=True)
        sorted_scores = scores[word_sort_ind]
        sorted_targets = targets[word_sort_ind]

        scores_upto_t = torch.zeros(batch_size, self.tagset_size)

        for t in range(max(sorted_lengths)):
            batch_size_t = sum([l > t for l in sorted_lengths])  # effective batch size (sans pads) at this timestep
            if t == 0:
                scores_upto_t[:batch_size_t] = sorted_scores[:batch_size_t, t, self.start_tag, :]  # (batch_size, tagset_size)
            else:
                # We add scores at current timestep to scores accumulated up to previous timestep, and log-sum-exp
                # Remember, the cur_tag of the previous timestep is the prev_tag of this timestep
                # So, broadcast prev. timestep's cur_tag scores along cur. timestep's cur_tag dimension
                scores_upto_t[:batch_size_t] = self._log_sum_exp(
                    sorted_scores[:batch_size_t, t, :, :] + scores_upto_t[:batch_size_t].unsqueeze(2), dim=1)  # (batch_size, tagset_size)

        # We only need the final accumulated scores at the <end> tag
        all_paths_scores = scores_upto_t[:, self.end_tag].sum()

        viterbi_loss = all_paths_scores - gold_score
        viterbi_loss = viterbi_loss / batch_size

        return viterbi_loss

    def _log_sum_exp(self, tensor, dim):
        """
        Calculates the log-sum-exponent of a tensor's dimension in a numerically stable way.

        :param tensor: tensor
        :param dim: dimension to calculate log-sum-exp of
        :return: log-sum-exp
        """
        m, _ = torch.max(tensor, dim)
        m_expanded = m.unsqueeze(dim).expand_as(tensor)
        return m + torch.log(torch.sum(torch.exp(tensor - m_expanded), dim))


