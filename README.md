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
* file_path: 
* num_training_records: 
* save_source_tokenizer_path: 
* save_target_tokenizer_path: 
* source_tokenizer_name: 
* target_tokenizer_name: 
* lstm_n_units: 
* train_batch_size: 
* save_model_path: 
* evaluation_percent: 
* num_epochs: 
```
However you can run train script with default values (after downloading required files and models and placing them in right places) by simply running `python train.py` . If you have installed `tensorboard` on your machine, you can check the
training procedure by accessing this `model/graph` file during training service. 

### Test your model
You can test your trained StringToString translator by running `test.py` script. Run `python test.py -h` to see options.
```
* --model_path: 
* --test_sentences: 
* --source_tokenizer: 
* --target_tokenizer: 
```
However you can run test script with default values (after downloading required files and models and placing them in right places) by simply running `python test.py` 

## Sample outputs
Here are some outputs for 

## Acknowledgements

Thanks to [Jason Brownlee](https://machinelearningmastery.com/develop-neural-machine-translation-system-keras/) article which inspired me to create and develop this repository.