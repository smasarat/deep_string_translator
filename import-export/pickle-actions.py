import pickle
from logconfig import log_config
import logging
from logging.config import dictConfig

dictConfig(log_config.LOGGING)

logger = logging.getLogger("file")


class PickleActions(object):
    def __init__(self, target_dir):
        self.target_dir = target_dir

    def save_pickle(self, content, file_name):
        try:
            pickle.dump(content, open("{}/{}".format(self.target_dir, file_name), 'wb'))
        except Exception as e:
            logger.exception(e)

    def load_pickle(self, file_name):
        try:
            return pickle.load(open("{}/{}".format(self.target_dir, file_name), 'rb'))
        except Exception as e:
            logger.exception(e)


def test_save_pickle():
    _pickle_actions = PickleActions(target_dir="../data")
    source_text_list = [["this is first line"], ["this is second line"]]
    _pickle_actions.save_pickle(content=source_text_list, file_name="source_list.pkl")


def test_load_pickle():
    _pickle_actions = PickleActions(target_dir="../data")
    tmp_pickle = _pickle_actions.load_pickle(file_name="source_list.pkl")
    print(len(tmp_pickle), tmp_pickle[0])
