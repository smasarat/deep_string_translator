import argparse

from keras import models
from numpy import unicode

from training.tokenize import SentencesTokenizer

parser = argparse.ArgumentParser(description='Load model and Test an input')
parser.add_argument('--model_path', type=str,
                    help='path to trained model (default=./model/ger_eng_model.h5)',
                    default="./model/ger_eng_model.h5")

parser.add_argument('--test_sentences', type=str,
                    help='comma separated test sentences',
                    default="ich bin brillentrager,du hast mir gefehlt")

parser.add_argument("--source_tokenizer_path", type=str,
                    help="path to source_tokenizer (default=./model/ger_tokenizer.pkl)",
                    default="./model/ger_tokenizer.pkl")

parser.add_argument("--target_tokenizer_path", type=str,
                    help="path to target_tokenizer (default=./model/eng_tokenizer.pkl)",
                    default="./model/eng_tokenizer.pkl")

args = parser.parse_args()

print(args.model_path)
model = models.load_model(args.model_path)
sentences_to_check_model = list(args.test_sentences.split(","))
source_tokenizer_path = args.source_tokenizer_path
target_tokenizer_path = args.target_tokenizer_path

_source_tokenizer = SentencesTokenizer()
_source_tokenizer.load_tokenizer(source_tokenizer_path)

_target_tokenizer = SentencesTokenizer()
_target_tokenizer.load_tokenizer(target_tokenizer_path)

list_sentences_to_check_model = _source_tokenizer.encode_sequences(sentences_to_check_model, training=False,
                                                                   padding_size=model.input_shape[1])

translated_sentences = model.predict(list_sentences_to_check_model)

for i in range(len(list_sentences_to_check_model)):
    print("{}:{}".format(sentences_to_check_model[i],
                         _target_tokenizer.decode_sequence(predictions=translated_sentences[i])).encode("utf-8"))
