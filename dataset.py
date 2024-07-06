from torch.utils.data import Dataset
from utils_data import *

class POSDataset(Dataset):

    def __init__(self, data, len_padded_words, len_padded_char, word_map, char_map, tag_map, model_type):

        """
        :param data: train/validation data
        :param len_padded_words:  max word sequence length
        :param len_padded_char: max character sequence length
        :param word_map: word map for whole dataset
        :param char_map: charecter map for whole dataset
        :param tag_map: tag map
        :param model_type: 'vanilla', 'dual', 'dual-highway' or 'crf'
        """
        self.data = data
        self.len_padded_words = len_padded_words
        self.len_padded_char = len_padded_char
        self.word_map = word_map
        self.char_map = char_map
        self.tag_map = tag_map
        self.model_type = model_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        """
        :return: padded encoded words, padded encoded forward chars, padded encoded backward chars,
            padded forward character markers as [0, 0, 1, 0, 0, 1, 0 ....], padded backward character markers, padded encoded tags,
            word sequence lengths, char sequence lengths
        """

        wmaps, w_l = create_input_words(self.data, idx, self.word_map, self.len_padded_words)
        crf_tags, true_tags, t_l = create_input_tag(self.data, idx, self.tag_map, self.len_padded_words)
        char_maps_f, char_maps_b, char_makers_f, char_makers_b, sen_l = create_input_char(self.data, idx, self.char_map, self.len_padded_char)

        assert w_l == t_l, 'Words sentence length and tag sentence length are not same'

        if self.model_type == 'crf':
            return wmaps, char_maps_f, char_maps_b, char_makers_f, char_makers_b, crf_tags, true_tags, w_l

        elif self.model_type == 'dual' or self.model_type == 'dual-highway':
            return wmaps, char_maps_f, char_makers_f, true_tags, w_l
        elif self.model_type == 'vanilla':
            return wmaps, true_tags, w_l
        else:
            raise NameError(f'There is no {self.model_type} in model types')

