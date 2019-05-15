import re, string
from unicodedata import normalize
import pickle
import logging
from logging.config import dictConfig

from logconfig import log_config

dictConfig(log_config.LOGGING)

logger = logging.getLogger("file")


class Process(object):
    def __init__(self, file_path=None, source_text_list=None):
        self.file_path = file_path
        self.source_text_list = None

    def normalize_text(self, normalize_unicodes=True, to_lower_case_filter=True, only_printable_chars_filter=True,
                       no_digits_and_punctuation_filter=True):

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
                tmp_source_text_list.append("\t".join([word for word in tmp_line.split("\t") if word.rstrip().lstrip().replace(" ", "").isalpha()]))
            source_text_list = tmp_source_text_list

        self.source_text_list = list(map(lambda x: x.split("\t"), source_text_list))
        return self.source_text_list


def test_open_process():
    _process = Process(file_path="../data/deu.txt")


def test_normalize_text():
    _process = Process(file_path="../data/deu.txt")
    source_text_list = _process.normalize_text()
    for i in range(100):
        if len(source_text_list[i]) == 2:
            print('[%s] => [%s]' % (source_text_list[i][0], source_text_list[i][1]))
