**Conditional Random Fields for Part-or-Speech Tagging in Natural Language Processing**
-

This is a PyTorch Project to Sequence Labeling. 
Part-of-Speech (POS) tagging is a fundamental task in Natural Language Processing (NLP) that involves labeling each word in a sentence with its corresponding part of speech (e.g., noun, verb, adjective). In this project, we explore four different models to tackle the POS tagging problem. 
Each model builds upon the previous one, incorporating more advanced techniques and architectures to improve performance. 

All libraries versions on [requirements.txt](https://github.com/VolkhinD/POS/blob/main/requirements.txt) file

**Objective**
--
Part-or-Speech Tagging plays a crucial role in NLP applications.
One of the most effective approaches for POS tagging is Conditional Random Fields (CRF)
![img](https://github.com/VolkhinD/POS/blob/main/img/1.jpeg)


**Concepts**
--
**Word-level and Charecter-level LSTMs.**
 * RNNs operating on individual characters in a text are known to capture the underlying style and structure.

 **Conditional Random Fields.**
 * Discrete classifiers predict a class or label at a word. Conditional Random Fields (CRFs) can do you one better – they predict labels based on not just the word, but also the neighborhood.
  Which makes sense, because there are patterns in a sequence of entities or labels. 
  Without a CRF, I simply use a single linear layer to transform the output of the Bidirectional LSTM into scores for each tag. 
  These are known as emission scores, which are a representation of the likelihood of the word being a certain tag. A CRF calculates not only the emission scores but also the transition scores, 
  which are the likelihood of a word being a certain tag considering the previous word was a certain tag.
  Therefore the transition scores measure how likely it is to transition from one tag to another.
  In the model, the CRF layer outputs the aggregate of the emission and transition scores at each word.

 **Viterbi Loss and Viterbi Decoding.**
  
 * Since I'm using CRFs, we're not so much predicting the right label at each word as we are predicting the right label sequence for a word sequence. 
  Viterbi Decoding is a way to do exactly this – find the most optimal tag sequence from the scores computed by a Conditional Random Field.

 **Highway Networks.**
  
 * Fully connected layers are a staple in any neural network to transform or extract features at different locations. 
  Highway Networks accomplish this, but also allow information to flow unimpeded across transformations. 
  This makes deep networks much more efficient or feasible. 
  A Highway Network is similar to a residual network, but a sigmoid-activated gate is use to determine the ratio in which the input and transformed output is combined.
  Since the character-RNNs contribute towards multiple tasks, Highway Networks are used for extracting task-specific information from its outputs.

 **Pretrained Embeddings: GloVe**
  * I use GloVe (Global Vectors for Word Representation) embeddings to initialize the word representations. GloVe embeddings are pretrained on large corpora, capturing semantic relationships      between words, which helps improve the performance of our models by providing rich word representations from the start.


**Models**
--
**Simplest Model: Word-Level bi-LSTM**

* Architecture: This model uses a bidirectional Long Short-Term Memory (bi-LSTM) network at the word level. The bi-LSTM processes each word in the sentence in both forward and backward directions, capturing contextual information from both sides.
Serves as the baseline model to compare the performance of more complex architectures.

**Dual Model: Character-Level and Word-Level LSTM**
* Architecture: This model combines character-level and word-level LSTMs. The character-level LSTM captures subword information by processing the sequence of characters for each word, while the word-level LSTM processes the sequence of words.
Aims to capture morphological features of words and improve the representation of rare or unseen words.

**Dual Model with Highway NN: Character-Level and Word-Level LSTM**
* Architecture: Building on the previous model, this version incorporates Highway Networks (Highway NN) to allow for better gradient flow and more flexible information transformation. Highway Networks are applied between the character-level and word-level LSTMs.
 Enhances the model's ability to learn complex patterns by allowing certain layers to carry forward information more effectively.

**CRF Model with Highway NN: Character-Level and Word-Level LSTM and CRF**

* Architecture: This is the most advanced model, which adds a Conditional Random Field (CRF) layer to word-level bidirectional LSTM combined with forward and backward character LSTM. ![IMG](https://github.com/VolkhinD/POS/blob/main/img/model.jpg)

The CRF layer considers the sequence of tags and captures dependencies between them, making it suitable for sequence labeling tasks like POS tagging.
Further improves the model's accuracy by considering the relationships between previous and current tags.

**Inputs to model**
--
I use data from NLTK brown corpus, categories='editorial'
There are eight inputs to models
For Viterbi Algorithm purpose I have to have start tag in a tag set, but it's not used for encodding sentence
But end tag must be in an encoded sentence. Also, in order to have the same length for all sentences I added \<pad> tag

**Words**

Word sequence must be tagged 
Original sentence is a list of tuples word: tag

[('Assembly', 'NOUN'), ('session', 'NOUN'), ('brought', 'VERB'), ('much', 'ADJ'), ('good', 'NOUN')]

So input will be:
* 'assembly', 'session', 'brought', 'much', 'good',  \<end>, \<pad>, \<pad> ...

Also, I create a word to index map which is an index mapping for each word in the corpus, including the <end>, and <pad> tokens.
PyTorch, like other libraries, needs words encoded as indices to look up embeddings for them. 
* [1, 2, 3, 4, 5, 8707, 0, 0, 0.....

Input to model is torchtext.vocab.vocab.Vocab, base class for all neural network modules. 
Since I'm using GloVe vectors for transfer learning (by preloading our embedding) not all words are able to transfer
from GloVe to vocab. That words are initialized randomly except \<pad> which initialized with only 0's
Number of out-of-corpus words: 299 out of 8708. 
They all can be seen if you call function <mark>test_embeds()</mark> in [test_data_utils.py](https://github.com/VolkhinD/POS/blob/main/test_data_utils.py)

**Characters (Forward)**

These are the character sequences in the forward direction.
* 'a', 's', 's', 'e', 'm', 'b', 'l', 'y', 's', 'e', 's', 's', 'i', 'o', 'n', 'b', .... \<end>, \<pad>, \<pad> ...

Note that there is no spase between words
And encode them with a char2in 
* [ 1, 19, 19,  5, 13,  2, 12, 25, 19,  5, 19, 19,  9, 15, 14,  2, 18, 15,
        21,  7,  8, 20, 13, 21,  3,  8,  7, 15, 15,  4, 27,  0,  0,  0,  0,  0,...

**Characters (Backward)**

This would be processed the same as the forward sequence, but backward.
(The <end> tag would still be at the end)

* 'd', 'o', 'o', 'g', 'h', 'c', 'u', 'm'...  \<end>, \<pad>, \<pad> ...

* [ 4, 15, 15,  7,  8,  3, 21, 13, 20,  8,  7, 21, 15, 18,  2, 14, 15,  9,
        19, 19,  5, 19, 25, 12,  2, 13,  5, 19, 19,  1, 27,  0,  0,  0,  0,...

**Character Makers**

Tensors for forward and backward character sequence that contain only 0 and 1 
and show the places of ends of words and \<end> tag

For forward 

* [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0,

**Tags**

For first 3 models tags labeled as usual

 * 'NOUN',  'NOUN', 'VERB', 'ADJ', 'NOUN', \<end>, \<pad>, \<pad> ...

After tag2idx

* [4, 4, 11, 10,  4, 13,  ...

For **CRF model** tags are made differently since previous tag must be included

But the outputs of the CRF layer are 2D m, m tensors at each word. 
To encode tag positions in these 2D outputs include previous tag 

(12, 4), (4, 4), (4, 11), (11, 10), (10, 4), (4, 13)

To unroll these scores to 1D tensor do 

* tag2in[prev_tag] * len(tag2in) + tag2in[curr_tag]

To get if \<start tag> is 14 in tag2in

* [172,  60,  67, 164, 144,  69,   0,   0, ...

**Lengths**

Length of sentence including \<end> tag

**Data Pipeline**
--
All data work at [utils_data.py](https://github.com/VolkhinD/POS/blob/main/utils_data.py)
<mark>create_maps()</mark> is function that returns word, tag and character maps. Called ones in [model_inputs.py](https://github.com/VolkhinD/POS/blob/main/model_inputs.py)

In PyTorch Dataset [in dataset.py](https://github.com/VolkhinD/POS/blob/main/dataset.py) other functions are called to create sequences for each sentence. 
Depending on model type, which are 'vanilla', 'dual', 'dual-highway' and 'crf' it returns the different outputs.

The Dataset is used by a PyTorch DataLoader in [train.py](https://github.com/VolkhinD/crf_for_pos/blob/main/train.py) to create and feed batches of data to the model for training or validation.

**Loss**
--
I use nn.NLLLoss() and nn.CrossEntropyLoss() for first 3 models
Since tag set contains \<end>, \<pad> and \<start>  tags they shouldn't contribute to the overall loss
So weight parameter [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]

For CRF model it's a custom Viterbi loss written in [models.py](https://github.com/VolkhinD/crf_for_pos/blob/main/models.py) To decode model outputs size
(batch_size, padded_sentence_length, tagset_size, tagset_size) Viterbi decoder is written in [decoder.py](https://github.com/VolkhinD/crf_for_pos/blob/main/decoder.py)
And toy example in <mark>toy_example_decoder()</mark> in [test_data_utils.py](https://github.com/VolkhinD/crf_for_pos/blob/main/test_data_utils.py)

**Training**
--
 See [train.py](https://github.com/VolkhinD/POS/blob/main/train.py)

To train your models from scratch, simply run <mark>train_all_models().</mark>
The result is python dictionary that contains lists of training / validation accuracies and
training / validation losses

I did training 3 times:
* Train models for 20 epoches and with no learning rate schedular
* Train models for 30 epoches and added StepLR schedular that reduces lr by 2 times every 5 epoches
![i](https://github.com/VolkhinD/POS/blob/main/img/Step.jpg)
*  Train models for 30 epoches and added CyclicLR schedular
  ![i](https://github.com/VolkhinD/POS/blob/main/img/Cycle.jpg)

