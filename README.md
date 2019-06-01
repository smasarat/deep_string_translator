# Deep StringToString Translator

This is an translator captioning codebase in Tensorflow and Keras. The model is created based on LSTM.
You can use pretrained model or train the new model yourself. If you have any idea to enhance the accuracy of model or you have any idea to change the model,
it would be my pleasure to discuss about them. Feel free to share it by sending me an [email](saman.masarat@gmail.com).  

## Requirements
`requirements.txt` contains whole of dependencies which are needed to run and develop this repository. Just run `pip install -r requirements.txt` in your virtual environment. 

Directory `data/` contains sample dataset which can be used for training model. You can also use your own data. The only assumption should be considered is it should be tab separated. The first column is source language text and the second column is target language text. See `data/deu_eng.txt`.

## Pretrained models
Pretrained models are provided [here]() for Dutch to English translation; and [here]() for English to Persian. It is obvious that the accuracy can be developed by considering more input data.

## Quick overview with Jupyter NoteBook
This [Jupyter Notebook](http://mscoco.org/dataset/) represents the whole process includes Loading data, tokenizing, encoding, normalizing, training, evaluation and testing step by step.

### Train your model
You can train this StringToString translator on any languages by running `train.py` script. Run `python train.py -h` to see options.
```
* --file_path: The path you have inserted your training data.
* --num_training_records: Define number of records for trainin (consider your Ram and cpy/gpu supports)
* --save_source_tokenizer_path: Path you want save your source tokenizer. You need it for testing procedure. 
* --save_target_tokenizer_path: Path you want save your target tokenizer. You need it for testing procedure.
* --lstm_n_units: Number of LSTM units in training process. 
* --train_batch_size: Number of batch elements should be passed to 
* --save_model_path: Path to the direcotry you want save the model. You need it for testing procedure.
* --evaluation_percent: Percent of data you want use for evaluation.
* --num_epochs: Number of epochs for training process.
```
You can run train script with default values (after downloading required files and models and placing them in right places) by simply running `python train.py` . If you have installed `tensorboard` on your machine, you can check the
training procedure by accessing this `model/graph` file during training service. However here is all you need to run train
script and save your model.

```
python train.py --file_path=./data/deu_eng.txt --num_training_records=10000 --save_source_tokenizer_path=./model/ger_tokenizer.pkl --save_target_tokenizer_path=./model/eng_tokenizer.pkl --ls
tm_n_units=256 --train_batch_size=64 --save_model_path=./model/ger_eng_model.h5 --tensor_board_model_path=./model/graph --evaluation_percent=0.2 --num_epochs=10 --training_batch_size=64
```

### Test your model
You can test your trained StringToString translator by running `test.py` script. Run `python test.py -h` to see options.
```
* --model_path: Path to model.
* --test_sentences: Place the test sentences in "". i.e "es ist zu spat,du hast mir gefehlt" 
* --source_tokenizer: Path to soruce tokenizer pickle file.
* --target_tokenizer: Path to target tokenizer pickle file.
```
You can run test script with default values (after downloading required files and models and placing them in right places) by simply running `python test.py`
However here is all you need to run test script and get outputs.
```
python test.py --model_path=./model/ger_eng_model.h5 --test_sentences="es ist zu spat,du hast mir gefehlt" --source_tokenizer=./model/ger_tokenizer.pkl --target_tokenizer=./model/eng_tokenizer.pkl
```

## Sample outputs
Here are some outputs for German to English: 


## Tensorboard outputs


## Acknowledgements

Thanks to [Jason Brownlee](https://machinelearningmastery.com/develop-neural-machine-translation-system-keras/) article which inspired me to create and develop this repository.