from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from numpy import array, argmax
from imports_exports.pickle_actions import PickleActions


class SentencesTokenizer(object):
    def __init__(self, num_words=None):
        self._tokenizer = Tokenizer(num_words=num_words)
        self.vocab_size = None
        self.length = None

    def create_tokenizer(self, text):
        """

        :param text: list of list .ie. [[first,sample],[second,sample]]
        :return:
        """
        self._tokenizer.fit_on_texts(text)
        self.vocab_size = len(self._tokenizer.word_index) + 1

    def encode_sequences(self, texts, training=True, padding_size=None):
        """

        :param padding_size: if model dimension is different from input do pad_sequence based on the padding_size
        :param texts: list of texts .ie. ["first example", "second sample"]
        :return: list of encoded input text with padding.
        """
        # integer encode sequences
        xx = self._tokenizer.texts_to_sequences(texts=texts)
        if training:
            self.length = max([len(sentence_iter.split()) for sentence_iter in texts])
            xx = pad_sequences(xx, maxlen=self.length, padding='post')
        if training is False:
            if padding_size is not None:
                xx = pad_sequences(xx, maxlen=padding_size, padding='post')
            else:
                xx = pad_sequences(xx, maxlen=self.length, padding='post')

        # pad sequences with 0 values

        return xx

    def encode_output(self, sequences):
        ylist = list()
        for sequence in sequences:
            encoded = to_categorical(sequence, num_classes=self.vocab_size)
            ylist.append(encoded)
        y = array(ylist)
        y = y.reshape(sequences.shape[0], sequences.shape[1], self.vocab_size)
        return y

    def decode_sequence(self, predictions):
        integers = [argmax(vector) for vector in predictions]
        target = list()
        word = None
        for i in integers:
            for word_iter, index in self._tokenizer.word_index.items():
                if index == i:
                    word = word_iter
                    break
                else:
                    word = None
            if word is None:
                break
            target.append(word)
        return ' '.join(target)

    def save_tokenizer(self, file_name="tokenizer", dir="../model"):
        _pickle_actions = PickleActions(target_dir=dir)
        _pickle_actions.save_pickle(content=self._tokenizer, file_name="{}.pkl".format(file_name))

    def load_tokenizer(self, file_name="tokenizer", dir="../model"):
        _pickle_actions = PickleActions(target_dir=dir)
        self._tokenizer = _pickle_actions.load_pickle(file_name="{}".format(file_name))
        self.length = len(self._tokenizer.word_index)


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
    xx = _sentence_tokenizer.encode_sequences(texts=test_text)
    print(xx)


def test_save_tokenizer():
    _source_tokenizer = SentencesTokenizer()
    samples = [["first", "sample"], ["second", "sample"]]
    _source_tokenizer.create_tokenizer(samples)
    _source_tokenizer.save_tokenizer(file_name="gholi_tokenizer")


def test_load_tokenizer():
    _source_tokenizer = SentencesTokenizer()
    print(_source_tokenizer.load_tokenizer("gholi_tokenizer"))
