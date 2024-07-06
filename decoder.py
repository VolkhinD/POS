import torch
class ViterbiDecoder():
    """
    Viterbi Decoder.
    https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Sequence-Labeling?tab=readme-ov-file#viterbi-decoding
    """

    def __init__(self, tag_map):
        """
        :param tag_map: tag map
        """
        self.tag_map = tag_map
        self.tagset_size = len(self.tag_map)
        self.start_tag = self.tag_map['<start>']
        self.end_tag = self.tag_map['<end>']

    def decode(self, scores, lengths):
        batch_size, word_pad_len, tag_size, _ = scores.size()

        assert self.tagset_size == tag_size, 'The size of tags set not the same as model returns'

        decoded = []

        for i in range(batch_size):
            viterbi_score = torch.zeros((self.tagset_size, lengths[i]))
            backpointer = torch.zeros((self.tagset_size), dtype=torch.long)
            viterbi_score[:, 0] = scores[i, 0, 0, :]
            for t in range(1, lengths[i]):
                backpointer[:] = viterbi_score[:, t - 1]
                max_scores, _ = torch.max(backpointer.unsqueeze(1).repeat(1, self.tagset_size) + scores[i, t, :, :], dim=0)
                viterbi_score[:, t] = max_scores
            _, max_idxs = torch.max(viterbi_score[1:-1, :], dim=0)
            max_idxs = torch.add(max_idxs, 1)
            max_idxs[lengths[i] - 1] = 5
            decoded.append(max_idxs)

        # add 0 to the end
        decoded = torch.stack(
            [torch.cat((decoded[i], torch.zeros((word_pad_len - lengths[i]))), dim=0) for i in range(batch_size)]
        )

        return decoded

