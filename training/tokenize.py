from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from numpy import array


class SentencesTokenizer(object):
    def __init__(self, num_words=None):
        self._tokenizer = Tokenizer(num_words=num_words)
        self.vocab_size = None

    def create_tokenizer(self, text):
        self._tokenizer.fit_on_texts(text)
        self.vocab_size = len(self._tokenizer.word_index) + 1

    def encode_sequences(self, length, texts):
        # integer encode sequences
        xx = self._tokenizer.texts_to_sequences(texts=texts)
        # pad sequences with 0 values
        xx = pad_sequences(xx, maxlen=length, padding='post')
        return xx

    def encode_output(self, sequences, vocab_size):
        ylist = list()
        for sequence in sequences:
            encoded = to_categorical(sequence, num_classes=self.vocab_size)
            ylist.append(encoded)
        y = array(ylist)
        y = y.reshape(sequences.shape[0], sequences.shape[1], self.vocab_size)
        return y


def test_create_tokenizer():
    _sentence_tokenizer = SentencesTokenizer()
    test_text = [
        ["this", "is", "first", "test"],
        ["this", "is", "second", "test"],
    ]
    _sentence_tokenizer.create_tokenizer(test_text)


def test_encode_sequences():
    _sentence_tokenizer = SentencesTokenizer()
    create_tokenizer_text = [
        ["this", "is", "first", "test"],
        ["this", "is", "second", "test"],
    ]

    _sentence_tokenizer.create_tokenizer(create_tokenizer_text)
    test_text = ["this is test", "this is second"]
    xx = _sentence_tokenizer.encode_sequences(length=max([len(sentence_iter) for sentence_iter in create_tokenizer_text]), texts=test_text)
    print(xx)
