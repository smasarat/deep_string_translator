from keras.layers import Embedding, LSTM, RepeatVector, TimeDistributed, Dense
from keras import Sequential
from sklearn.model_selection import train_test_split

from textprocessing.processing import Process
import argparse
from keras.callbacks import TensorBoard, ModelCheckpoint
from training.tokenize import SentencesTokenizer

parser = argparse.ArgumentParser(description='Load data and Train a Model')

parser.add_argument('--file_path', type=str,
                    help='tab-separated text file (default=./data/deu_to_eng.txt)',
                    default="./data/deu_to_eng.txt")

parser.add_argument('--ignore_all_normalizations', type=bool,
                    help='If you want to avoid any normalization, pass True to it. In some languages such as Persian, encoding problems may effect the normalization and '
                         'outputs. Pass it True if you are not sure about probable encoding problems.',
                    default=False)

parser.add_argument('--num_training_records', type=int,
                    help='Number of rows for training (default=10000) ',
                    default=1000)

parser.add_argument('--save_source_tokenizer_path', type=str,
                    help='Save source tokenizer in desired destination. We recommend save it. You need load it in test process.'
                         'Enter None for Skipping saving status default address (./model/source_tokenizer.pkl)',
                    default="./model/source_tokenizer.pkl")

parser.add_argument('--save_target_tokenizer_path', type=str,
                    help='Save target tokenizer in desired destination. We recommend save it. You need load it in test process.'
                         'Enter None for Skipping saving status default address (./model/target_tokenizer.pkl)',
                    default="./model/target_tokenizer.pkl")

parser.add_argument('--lstm_n_units', type=int,
                    help='number of units in LSTM training (default=256)',
                    default=256)

parser.add_argument('--train_batch_size', type=int, help='batch size for training (default=64)',
                    default=64)

parser.add_argument('--save_model_path', type=str,
                    help='allocate name to the model (default=./model/model.h5)',
                    default="./model/model.h5")

parser.add_argument('--tensor_board_model_path', type=str,
                    help='You can use tensorboard to visualize training procedure with this file)(default=./model/graph)',
                    default="'./model/graph'")

parser.add_argument('--evaluation_percent', type=float,
                    help='allocate the percent you want perform for validation (default=0.2)',
                    default=0.2)

parser.add_argument('--num_epochs', type=int,
                    help='Number of epochs for training (default=10)',
                    default=10)

parser.add_argument('--training_batch_size', type=int,
                    help='Number of epochs for training (default=64)',
                    default=64)

args = parser.parse_args()
file_path = args.file_path
ignore_all_normalizations = args.ignore_all_normalizations
number_of_records = args.num_training_records
save_source_tokenizer_path = args.save_source_tokenizer_path
save_target_tokenizer_path = args.save_target_tokenizer_path
model_name = args.save_model_path
evaluation_percent = args.evaluation_percent
tensor_board_model_path = args.tensor_board_model_path
training_batch_size = args.training_batch_size
num_epochs = args.num_epochs

# reading data and normalizing
_process = Process(file_path=file_path)
# In language such as Persian where the encoding may play considerable role in output, ignore normalization.
# In future versions more configurations will be provided. I will be thankful if you share any idea.
if ignore_all_normalizations:
    input_text_list = _process.normalize_text(normalize_unicodes=False, to_lower_case_filter=False,
                                              only_printable_chars_filter=False, no_digits_and_punctuation_filter=False,
                                              drop_abnormals=False)
else:
    input_text_list = _process.normalize_text()

input_text_list = input_text_list[:number_of_records]

################### SOURCE #######################
source_text_list = [item[1].split() for item in input_text_list]
# tokenizer
_source_tokenizer = SentencesTokenizer()
_source_tokenizer.create_tokenizer(source_text_list)
_source_tokenizer.save_tokenizer(file_path=save_source_tokenizer_path)

# encoding
x_vector = _source_tokenizer.encode_sequences(list(map(lambda x: " ".join(x), source_text_list)))

################### TARGET #######################

# creating destination normalizer
target_text_list = [item[0].split() for item in input_text_list]
_target_tokenizer = SentencesTokenizer()
_target_tokenizer.create_tokenizer(target_text_list)
_target_tokenizer.save_tokenizer(file_path=save_target_tokenizer_path)

# encoding
y_vector = _target_tokenizer.encode_sequences(list(map(lambda x: " ".join(x), target_text_list)))
y_vector = _target_tokenizer.encode_output(y_vector)

################## DEEP TRAINING ###############
model = Sequential()

n_units = 256
model.add(Embedding(input_dim=_source_tokenizer.vocab_size, output_dim=n_units,
                    input_length=_source_tokenizer.length, mask_zero=True))
model.add(LSTM(units=n_units))
model.add(RepeatVector(_target_tokenizer.length))
model.add(LSTM(n_units, return_sequences=True))
model.add(TimeDistributed(Dense(_target_tokenizer.vocab_size, activation='softmax')))

model.compile(optimizer='adam', loss='categorical_crossentropy')
print(model.summary())

# now you can access the TensorBoard panel (if you have installed tensorboard)
# with command tensorboard --logdir=:./model/graph
tensor_board_callback = TensorBoard(log_dir=tensor_board_model_path, histogram_freq=0, write_graph=True,
                                    write_images=True)
# save the model for further uses.
checkpoint = ModelCheckpoint(model_name, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

X_train, X_test, y_train, y_test = train_test_split(x_vector, y_vector, test_size=evaluation_percent, random_state=42)
print("Run tensorboard to see error rate visually")
model.fit(x=x_vector, y=y_vector, validation_data=(X_test, y_test), callbacks=[tensor_board_callback, checkpoint],
          batch_size=64, epochs=num_epochs, verbose=2)
