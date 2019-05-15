from keras import Sequential
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Embedding, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.contrib.learn.python.learn.estimators._sklearn import train_test_split

from textprocessing.processing import Process
from training.tokenize import SentencesTokenizer

number_of_records = 10000

# reading data and normalizing
_process = Process(file_path="data/deu.txt")
input_text_list = _process.normalize_text()
input_text_list = _process.drop_abnormals()
input_text_list = input_text_list[:number_of_records]

################### SOURCE #######################
source_text_list = [item[1].split() for item in input_text_list]
# tokenize
_source_tokenizer = SentencesTokenizer()
_source_tokenizer.create_tokenizer(source_text_list)

# encoding
x_vector = _source_tokenizer.encode_sequences(list(map(lambda x: " ".join(x), source_text_list)))

################### TARGET #######################

# creating destination normalizer
target_text_list = [item[0].split() for item in input_text_list]
_target_tokenizer = SentencesTokenizer()
_target_tokenizer.create_tokenizer(target_text_list)

# encoding
y_vector = _target_tokenizer.encode_sequences(list(map(lambda x: " ".join(x), target_text_list)))
y_vector = _target_tokenizer.encode_output(y_vector)

print()

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
tensor_board_callback = TensorBoard(log_dir='./model/graph', histogram_freq=0, write_graph=True, write_images=True)
# save the model for further uses.
checkpoint = ModelCheckpoint('./model/model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

X_train, X_test, y_train, y_test = train_test_split(x_vector, y_vector, test_size=0.2, random_state=42)
model.fit(x=x_vector, y=y_vector, validation_data=(X_test, y_test), callbacks=[tensor_board_callback, checkpoint],
          batch_size=64, epochs=10, verbose=2)
