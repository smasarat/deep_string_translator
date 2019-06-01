import re, string
from unicodedata import normalize
import pickle
import logging

from logconfig import log_config

logging.basicConfig(format=log_config.FORMAT)

logger = logging.getLogger("file")


class Process(object):
    def __init__(self, file_path=None, source_text_list=None):
        self.file_path = file_path
        self.source_text_list = None

    def normalize_text(self, normalize_unicodes=True, to_lower_case_filter=True, only_printable_chars_filter=True,
                       no_digits_and_punctuation_filter=True, drop_abnormals=True):
        """
        We developed the system based on the tab seperated format by default. You can easily change it
        the format we have worked on it is following below format:

            Hi.	Hallo!
            Hi.	Grüß Gott!
            Run!	Lauf!
            Wow!	Potzdonner!
            Wow!	Donnerwetter!

        First column represent source strings and the second column represents the target we want to get if we enter the
        first column to our model.

        :param normalize_unicodes: False if you do not want to perform unicode normalization
        :param to_lower_case_filter: False if you do not want to perform lower case filtering normalization
        :param only_printable_chars_filter: False if you do not want to perform only printable normalization
        :param no_digits_and_punctuation_filter: False if you do not want to perform only digits normalization
        :param drop_abnormals: if the repetition of the word is more than once, drop the extra ones !
        :return: list of lists. with normalization performed on them.
        """

        with open(file=self.file_path, encoding="utf-8", mode="rt") as s_f:
            source_file = s_f.read()
            source_text_list = source_file.rstrip().lstrip().split("\n")

        if normalize_unicodes:
            tmp_source_text_list = []
            for tmp_line in source_text_list:
                tmp_source_text_list.append(normalize('NFD', tmp_line).encode('ascii', 'ignore'))
            source_text_list = list(map(lambda x: x.decode("utf-8"), tmp_source_text_list))
        if to_lower_case_filter:
            source_text_list = list(map(lambda x: str(x).lower(), source_text_list))

        if only_printable_chars_filter:
            printable_filter = re.compile("[^%s]" % re.escape(string.printable))
            tmp_source_text_list = []
            for tmp_line in source_text_list:
                tmp_source_text_list.append([printable_filter.sub('', w) for w in tmp_line])
            source_text_list = list(map(lambda x: "".join(x), tmp_source_text_list))

        if no_digits_and_punctuation_filter:
            tmp_source_text_list = []
            table = str.maketrans('', '', string.punctuation)
            for tmp_line in source_text_list:
                tmp_source_text_list.append([word.translate(table) for word in tmp_line])

            source_text_list = list(map(lambda x: "".join(x), tmp_source_text_list))
            tmp_source_text_list = []
            for tmp_line in source_text_list:
                tmp_source_text_list.append("\t".join(
                    [word for word in tmp_line.split("\t") if word.rstrip().lstrip().replace(" ", "").isalpha()]))
            source_text_list = tmp_source_text_list

        self.source_text_list = list(map(lambda x: x.split("\t"), source_text_list))

        if drop_abnormals:
            self.source_text_list = list(filter(lambda x: len(x) == 2, self.source_text_list))

        return self.source_text_list


def test_open_process():
    _process = Process(file_path="../data/deu_to_eng.txt")


def test_normalize_text():
    _process = Process(file_path="../data/deu_to_eng.txt")
    source_text_list = _process.normalize_text()
    for i in range(100):
        if len(source_text_list[i]) == 2:
            print('[%s] => [%s]' % (source_text_list[i][0], source_text_list[i][1]))
